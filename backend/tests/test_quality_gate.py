"""CI quality gate: detection quality on the hand-annotated gold set
(evals/data/blacken_gold.txt) must not regress below these floors.

Floors sit a few points under the measured values (see evals/results/) so
that noise from a dependency bump doesn't flake the build, while a real
regression - a recognizer dropped, a threshold mis-set - fails loudly.
Run the full harness (`python -m evals.run_detection`) for the complete
report; this is the fast (~1s) subset that guards every commit.
"""

from evals.datasets import load_blacken_gold
from evals.metrics import aggregate, score_document, select_predictions
from evals.run_detection import to_spans

FLOORS = {
    "token_recall_auto": 0.88,
    "token_precision_auto": 0.90,
    "entity_recall_auto": 0.86,
    "negative_false_alarm_rate_auto_max": 0.30,
}


def test_gold_set_quality_gate(engine):
    docs = []
    for sample in load_blacken_gold():
        entities = engine.detect(sample.text)
        pred = to_spans(select_predictions(entities, "auto"))
        docs.append(score_document(sample.text, sample.spans, pred))
    r = aggregate(docs)

    report = (
        f"token R={r['token']['recall']:.3f} P={r['token']['precision']:.3f} "
        f"entity R={r['overlap']['recall']:.3f} negFA={r['negative_false_alarm_rate']:.3f}\n"
        f"top misses: {r['top_misses'][:8]}"
    )
    assert r["token"]["recall"] >= FLOORS["token_recall_auto"], report
    assert r["token"]["precision"] >= FLOORS["token_precision_auto"], report
    assert r["overlap"]["recall"] >= FLOORS["entity_recall_auto"], report
    assert r["negative_false_alarm_rate"] <= FLOORS["negative_false_alarm_rate_auto_max"], report
