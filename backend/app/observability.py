"""Observability: request IDs, structured access logs, Prometheus metrics,
and optional Sentry - designed so that no document content or PII is ever
emitted.

* Every request gets an X-Request-ID (honoured if the client sends one),
  echoed in the response and attached to its log line, so a user-reported
  failure can be traced to one log entry.
* One access-log line per request: method, *route template* (e.g.
  /api/download/{document_id} - never the concrete path), status, latency.
  LOG_FORMAT=json emits JSON lines for log aggregators.
* /api/metrics serves Prometheus text format from a tiny in-process registry
  (no prometheus_client dependency): request counts and latency histograms per
  route, plus documents processed and entities detected per type.
* Sentry is initialised only if SENTRY_DSN is set and sentry-sdk is
  installed, with send_default_pii=False and request bodies/headers stripped.
"""

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("blacken.access")

_LATENCY_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = defaultdict(float)
        self.hist_buckets: Dict[Tuple[str, str], list] = {}
        self.hist_sum: Dict[Tuple[str, str], float] = defaultdict(float)
        self.hist_count: Dict[Tuple[str, str], int] = defaultdict(int)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe_latency(self, route: str, method: str, seconds: float) -> None:
        key = (route, method)
        with self._lock:
            buckets = self.hist_buckets.setdefault(key, [0] * len(_LATENCY_BUCKETS_S))
            for i, bound in enumerate(_LATENCY_BUCKETS_S):
                if seconds <= bound:
                    buckets[i] += 1
            self.hist_sum[key] += seconds
            self.hist_count[key] += 1

    def render(self, gauges: Dict[str, float] = None) -> str:
        def fmt_labels(pairs):
            return "{" + ",".join(f'{k}="{v}"' for k, v in pairs) + "}" if pairs else ""

        lines = []
        with self._lock:
            seen = set()
            for (name, labels), value in sorted(self.counters.items()):
                if name not in seen:
                    lines.append(f"# TYPE {name} counter")
                    seen.add(name)
                lines.append(f"{name}{fmt_labels(labels)} {value:g}")
            lines.append("# TYPE blacken_request_duration_seconds histogram")
            for (route, method), buckets in sorted(self.hist_buckets.items()):
                base = [("method", method), ("route", route)]
                for bound, count in zip(_LATENCY_BUCKETS_S, buckets):
                    lines.append(
                        f"blacken_request_duration_seconds_bucket{fmt_labels(base + [('le', str(bound))])} {count}"
                    )
                lines.append(
                    f"blacken_request_duration_seconds_bucket{fmt_labels(base + [('le', '+Inf')])} {self.hist_count[(route, method)]}"
                )
                lines.append(f"blacken_request_duration_seconds_sum{fmt_labels(base)} {self.hist_sum[(route, method)]:.6f}")
                lines.append(f"blacken_request_duration_seconds_count{fmt_labels(base)} {self.hist_count[(route, method)]}")
        for name, value in (gauges or {}).items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value:g}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


def _route_template(request: Request) -> str:
    """The matched route's path template (/api/download/{document_id}), never
    the concrete URL - concrete paths carry document ids, which would both
    leak identifiers into logs and explode metric cardinality.

    Rebuilt from the URL by swapping each matched path-parameter segment for
    its {name}: version-independent, unlike reading the route object from
    scope (FastAPI >= 0.16x nests included routers, so that object's path
    lacks the /api prefix)."""
    if request.scope.get("endpoint") is None:
        return "unmatched"
    params = {str(v): k for k, v in (request.scope.get("path_params") or {}).items()}
    segments = request.url.path.split("/")
    return "/".join("{" + params[s] + "}" if s in params else s for s in segments)


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, json_logs: bool = False):
        super().__init__(app)
        self.json_logs = json_logs

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = _route_template(request)
            metrics.inc("blacken_requests_total", route=route, method=request.method, status=str(status))
            metrics.observe_latency(route, request.method, elapsed)
            record = {
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "status": status,
                "duration_ms": round(elapsed * 1000, 1),
            }
            if self.json_logs:
                logger.info(json.dumps(record))
            else:
                logger.info(
                    "%s %s %s %d %.1fms", request_id, request.method, route, status, elapsed * 1000
                )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str, json_logs: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def _scrub_event(event, _hint):
    """Never ship request bodies, headers, cookies or query strings: for this
    app any of them can carry document text, entities or owner tokens."""
    req = event.get("request") or {}
    for key in ("data", "headers", "cookies", "query_string", "env"):
        req.pop(key, None)
    return event


def init_sentry(dsn: str, environment: str) -> bool:
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logging.getLogger("blacken").warning("SENTRY_DSN set but sentry-sdk not installed")
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        send_default_pii=False,
        traces_sample_rate=0.0,
        before_send=_scrub_event,
        max_request_body_size="never",
    )
    return True
