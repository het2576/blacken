"""Optional transformer PII detector: GLiNER run on onnxruntime, no PyTorch.

GLiNER (knowledgator/gliner-pii-base-v1.0, Apache-2.0) is a span-based
model fine-tuned for PII: given the text and a list of label names, it scores
every span of up to `max_width` words against every label. It recognises
names, addresses and identifiers from context, where spaCy's general-purpose
NER and the regex recognizers have nothing to key on.

Why not the official `gliner` package: it imports PyTorch (~700 MB on disk,
hundreds of MB resident) - impossible on a 512 MB free-tier container. The
model ships an int8-quantised ONNX export, so this module re-implements
GLiNER's small pre/post-processing (word splitting, prompt construction,
span enumeration, greedy decoding) directly on `onnxruntime` + `tokenizers`.

Enabled only when PII_MODEL_DIR points at a directory holding
`model_quint8.onnx` (or `model.onnx`), `tokenizer.json` and
`gliner_config.json`; otherwise Blacken runs exactly as before.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from presidio_analyzer import EntityRecognizer, RecognizerResult

logger = logging.getLogger("blacken")

# GLiNER label prompt -> Blacken entity type. Labels follow the model card's
# predefined PII categories. Countries/states, money and medical conditions
# are deliberately not requested: not identifying, and they cost compute.
LABEL_TO_ENTITY: Dict[str, str] = {
    "name": "PERSON",
    "location address": "ADDRESS",
    "location street": "ADDRESS",
    "location city": "LOCATION",
    "location zip": "ZIP_CODE",
    "dob": "DATE_TIME",
    "email address": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "ssn": "NATIONAL_ID",
    "passport number": "PASSPORT_NUMBER",
    "driver license": "DRIVER_LICENSE",
    "account number": "ACCOUNT_NUMBER",
    "routing number": "ROUTING_NUMBER",
    "credit card": "CREDIT_CARD",
    "healthcare number": "MEDICAL_RECORD_NUMBER",
    "ip address": "IP_ADDRESS",
}

_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")  # GLiNER's WhitespaceTokenSplitter


class GlinerOnnxRecognizer(EntityRecognizer):
    """Presidio recognizer wrapping an int8 GLiNER ONNX model."""

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.4,
        window_words: int = 192,
        stride_words: int = 160,
        labels: Optional[Sequence[str]] = None,
        supported_language: str = "en",
    ):
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.window_words = window_words
        self.stride_words = stride_words
        self.labels = list(labels or LABEL_TO_ENTITY)
        super().__init__(
            supported_entities=sorted({LABEL_TO_ENTITY[l] for l in self.labels}),
            name="GlinerOnnxRecognizer",
            supported_language=supported_language,
        )

    def load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = next(
            (self.model_dir / n for n in ("model_quint8.onnx", "model_int8.onnx", "model.onnx")
             if (self.model_dir / n).exists()),
            None,
        )
        if model_path is None:
            raise FileNotFoundError(f"No GLiNER ONNX model found in {self.model_dir}")
        config = json.loads((self.model_dir / "gliner_config.json").read_text())
        self.max_width = int(config.get("max_width", 12))
        # "markerV0" = span head (base model); "token_level" = start/end/inside
        # head (small and edge models). Both decode to the same spans.
        self.span_mode = config.get("span_mode", "markerV0")
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.ent_token = config.get("ent_token", "<<ENT>>")
        self.sep_token = config.get("sep_token", "<<SEP>>")

        opts = ort.SessionOptions()
        # The default CPU memory arena keeps peak-sized buffers alive between
        # requests; on a 512 MB container steady-state RSS matters more than
        # the small allocation speed-up the arena buys.
        opts.enable_cpu_mem_arena = False
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.prompt = []
        for label in self.labels:
            self.prompt += [self.ent_token, label]
        self.prompt.append(self.sep_token)
        logger.info("GLiNER PII model loaded from %s (%d labels)", model_path.name, len(self.labels))

    # ------------------------------------------------------------ inference
    def _predict_window(self, words: List[str]) -> List[Tuple[int, int, int, float]]:
        """Returns (start_word, end_word_inclusive, label_idx, prob) for one
        window, before cross-label overlap resolution."""
        n = len(words)
        enc = self.tokenizer.encode(self.prompt + words, is_pretokenized=True)
        prompt_len = len(self.prompt)
        words_mask, prev = [], None
        for wid in enc.word_ids:
            if wid is None or wid < prompt_len or wid == prev:
                words_mask.append(0)
            else:
                words_mask.append(wid - prompt_len + 1)
            prev = wid

        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "words_mask": np.array([words_mask], dtype=np.int64),
            "text_lengths": np.array([[n]], dtype=np.int64),
        }
        if self.span_mode == "token_level":
            return self._decode_token_level(self.session.run(None, feeds)[0][0])

        span_idx = np.array(
            [[i, i + k] for i in range(n) for k in range(self.max_width)], dtype=np.int64
        )
        span_mask = span_idx[:, 1] < n
        feeds["span_idx"] = (span_idx * span_mask[:, None])[None, :, :]
        feeds["span_mask"] = span_mask[None, :]
        logits = self.session.run(None, feeds)[0][0]  # (n, max_width, num_labels)
        probs = 1.0 / (1.0 + np.exp(-logits))
        out = []
        for s, k, c in zip(*np.where(probs > self.threshold)):
            if s + k < n:
                out.append((int(s), int(s + k), int(c), float(probs[s, k, c])))
        return out

    def _decode_token_level(self, logits: np.ndarray) -> List[Tuple[int, int, int, float]]:
        """logits: (n_words, n_labels, 3) = start / end / inside scores.
        A span (s, e, label) is valid when start[s] and end[e] clear the
        threshold and every word inside does too; its score is the minimum of
        those probabilities (GLiNER's TokenDecoder semantics)."""
        probs = 1.0 / (1.0 + np.exp(-logits))
        th = self.threshold
        out = []
        for c in range(probs.shape[1]):
            starts = np.where(probs[:, c, 0] > th)[0]
            ends = np.where(probs[:, c, 1] > th)[0]
            for s in starts:
                for e in ends:
                    if e < s or e - s >= self.max_width:
                        continue
                    inside = probs[s : e + 1, c, 2]
                    if (inside < th).any():
                        continue
                    score = float(min(inside.min(), probs[s, c, 0], probs[e, c, 1]))
                    out.append((int(s), int(e), c, score))
        return out

    def predict(self, text: str) -> List[Tuple[int, int, str, float]]:
        """Character spans (start, end, entity_type, score) over the full
        text, using overlapping word windows so long documents fit."""
        matches = list(_WORD.finditer(text))
        if not matches:
            return []
        words = [m.group() for m in matches]
        candidates: Dict[Tuple[int, int, int], float] = {}
        start = 0
        while True:
            window = words[start : start + self.window_words]
            for s, e, c, p in self._predict_window(window):
                key = (start + s, start + e, c)
                candidates[key] = max(p, candidates.get(key, 0.0))
            if start + self.window_words >= len(words):
                break
            start += self.stride_words

        # GLiNER's flat greedy decoding: highest score first, no overlaps.
        chosen: List[Tuple[int, int, int, float]] = []
        for (s, e, c), p in sorted(candidates.items(), key=lambda kv: -kv[1]):
            if any(not (e < cs or s > ce) for cs, ce, _c, _p in chosen):
                continue
            chosen.append((s, e, c, p))
        return [
            (matches[s].start(), matches[e].end(), LABEL_TO_ENTITY[self.labels[c]], p)
            for s, e, c, p in sorted(chosen)
        ]

    def analyze(self, text, entities, nlp_artifacts=None) -> List[RecognizerResult]:
        wanted = set(entities or self.supported_entities)
        results = []
        for start, end, entity, score in self.predict(text):
            if entity not in wanted:
                continue
            results.append(
                RecognizerResult(
                    entity_type=entity,
                    start=start,
                    end=end,
                    score=round(score, 4),
                    analysis_explanation=None,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )
        return results
