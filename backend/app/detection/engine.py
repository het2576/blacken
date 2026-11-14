"""Presidio-based PII detection engine.

Wraps presidio-analyzer's AnalyzerEngine with a spaCy NLP backend plus a
handful of custom pattern recognizers, and normalizes results into the
same `{text, type, start, end, confidence, source, selected}` shape the
frontend already expects (see src/services/api.ts `Entity`).

Presidio already handles what the previous hand-rolled engine tried (and
mostly failed) to do itself: overlap/duplicate resolution between
recognizers, and confidence boosting from nearby context words. We only
add a light per-type minimum-score filter on top, in one place.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import spacy

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    AbaRoutingRecognizer,
    InAadhaarRecognizer,
    InGstinRecognizer,
    InPanRecognizer,
    InPassportRecognizer,
    InVehicleRegistrationRecognizer,
    InVoterRecognizer,
    UkNinoRecognizer,
    UkPostcodeRecognizer,
)

from app.detection.gazetteers import is_non_identifying_place
from app.detection.recognizers import ALL_CUSTOM_RECOGNIZERS

logger = logging.getLogger("blacken")

# Presidio ships these checksum/format-validated recognizers but does NOT
# register them in its default registry, so before this list existed an
# Aadhaar, PAN, GSTIN, UK National Insurance number or UK postcode was never
# detected at all - despite the README claiming coverage. Each was checked
# against the dev split for precision before being enabled; noisier ones
# (generic 9-digit UK passport, US NPI, AU/CA IDs) stay off until measured.
EXTRA_PREDEFINED_RECOGNIZERS = (
    InAadhaarRecognizer,
    InPanRecognizer,
    InGstinRecognizer,
    InPassportRecognizer,
    InVoterRecognizer,
    InVehicleRegistrationRecognizer,
    UkNinoRecognizer,
    UkPostcodeRecognizer,
    AbaRoutingRecognizer,
)

# Deliberately no entity allowlist here: this is a general personal-data
# tool used on real documents from anywhere, not just US-formatted ones.
# Note that Presidio's *default* registry only loads a subset of its
# country recognizers (US_SSN, UK_NHS, ...); India/UK formats such as
# IN_AADHAAR, IN_PAN and UK_NINO must be registered explicitly - see
# EXTRA_PREDEFINED_RECOGNIZERS. The rare cost is a cosmetic one - two country
# recognizers can occasionally tie on an ambiguous digit string and the
# entity gets the "wrong" country label - but it still gets flagged and
# redacted either way, which is what actually matters.
MIN_SCORE_BY_TYPE: Dict[str, float] = {
    # Lowered from spaCy's flat 0.85 floor to 0.4 so structurally-downweighted
    # PERSON guesses (see _adjust_person_confidence) stay visible for review
    # instead of disappearing entirely.
    "PERSON": 0.4,
    # spaCy's ORG tag on this pipeline gives every hit the same flat, low
    # confidence (0.85 x the 0.4 multiplier below = 0.34) regardless of
    # whether it's actually a company name, and it is noisy on real
    # documents: job titles ("AI/ML Intern", "Project Mentor & Director"),
    # address fragments ("Silverpark Soc", "Pal Rd"), department names,
    # bare acronyms ("AI").
    #
    # This sat at 0.4 - just *above* that flat 0.34 - which dropped every
    # spaCy org guess before the user ever saw it. Measured on a 12-document
    # set, that cost 7 of 35 expected entities, and the losses were not only
    # companies: spaCy mislabels some personal names as ORG ("Bjorn
    # Haraldsson"), so suppressing the label silently leaked a real name into
    # "redacted" output. For a tool whose failure mode is exposing PII, a
    # noisy entity the user can untick beats a missing one they never see.
    #
    # 0.3 is below the flat score, so these now surface - but they stay well
    # under AUTO_SELECT_MIN_SCORE (0.6), so they are shown for review and
    # never pre-checked for redaction. Measured effect: recall 28/35 -> 35/35
    # with the auto-selected count unchanged at 27, at the cost of 2 extra
    # unchecked false positives on a deliberately noisy sample.
    "ORGANIZATION": 0.3,
    "LOCATION": 0.5,
    "NRP": 0.5,
    # Lowered analogously to PERSON so structurally-downweighted false
    # positives (see _adjust_date_confidence) stay visible for review.
    "DATE_TIME": 0.3,
    "AGE": 0.6,
}
DEFAULT_MIN_SCORE = 0.35

# Whether an entity defaults to checked/selected for redaction. This is
# deliberately a *separate*, higher bar than MIN_SCORE_BY_TYPE above: we
# still want to *show* a low-confidence guess for the user to review, but
# a low-confidence guess should not be pre-checked for a destructive
# action (blacking out text) - the user should opt in, not opt out.
AUTO_SELECT_MIN_SCORE = 0.6

# Optional per-type overrides fitted on dev data by evals/calibrate.py. Kept
# out of code so a re-fit is a reviewed data change, and deleting the file
# reverts to the single global threshold above.
CALIBRATED_THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.json"


def _load_calibrated_thresholds() -> Dict[str, float]:
    try:
        data = json.loads(CALIBRATED_THRESHOLDS_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable %s", CALIBRATED_THRESHOLDS_PATH.name)
        return {}
    return {str(k): float(v) for k, v in (data.get("per_type") or {}).items()}

# Real full names in documents are overwhelmingly multi-word and every
# word is capitalized ("Het Limbachiya"). A single bare capitalized word
# spaCy tags PERSON is, in practice on real documents, at least as likely
# to be a product/tool/brand name picked up out of a skills list or tech
# stack ("Docker", "Streamlit", "Claude") as an actual first-name-only
# mention - spaCy's NER gives every hit the same flat confidence
# regardless, so there's no score-based way to tell them apart otherwise.
# This down-weights (not drops) those cases: still detected and shown,
# just not pre-selected, and short ALL-CAPS tokens (likely acronyms, e.g.
# "API", "NLP", "CI") get the same treatment even inside a multi-word span.
PERSON_CONFIDENCE_PENALTY = 0.5

# Recognizers whose PERSON output already reflects context (a "Name:" label,
# or a transformer reading the whole sentence), so the bare-single-word
# penalty - which exists because spaCy tags "Docker" PERSON - doesn't apply.
CONTEXTUAL_PERSON_RECOGNIZERS = {"LabeledPersonRecognizer", "GlinerOnnxRecognizer"}


def _looks_like_a_name(entity_text: str) -> bool:
    words = entity_text.split()
    if len(words) < 2:
        return False
    for word in words:
        if not word[:1].isupper():
            return False
        if word.isupper() and len(word) <= 5 and word.isalpha():
            return False  # short ALL-CAPS token - likely an acronym, not a name
    return True


# Contexts that make a single capitalised word a person rather than a tool
# or brand: an honorific right before it ("Mr. Hayes"), a greeting or
# sign-off ("Dear Sam", "Thanks Rahul", "Cheers,\nMegan"), or a possessive
# ("Julia's place"). These are closed, document-independent vocabularies.
_HONORIFIC_BEFORE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Madam|Shri|Smt|Sri|Officer|Agent|Nurse)\.?\s+$"
)
_GREETING_BEFORE = re.compile(
    r"(?:^|\b)(?:Hi|Hello|Hey|Dear|Thanks|Thank\s+you|Regards|Best|Cheers|Sincerely|"
    r"Warmly|Yours|Love|Bye)[\s,!]+$",
    re.IGNORECASE,
)
_POSSESSIVE_AFTER = re.compile(r"^'s\b")


def _adjust_person_confidence(
    entity_text: str,
    score: float,
    text: str = "",
    start: int = 0,
    end: int = 0,
    known_name_parts: Set[str] = frozenset(),
) -> float:
    if _looks_like_a_name(entity_text):
        return score
    word = entity_text.strip()
    if word[:1].isupper() and not (word.isupper() and len(word) <= 5):
        before = text[max(0, start - 24) : start]
        after = text[end : end + 3]
        if (
            _HONORIFIC_BEFORE.search(before)
            or _GREETING_BEFORE.search(before)
            or _POSSESSIVE_AFTER.match(after)
            # Same document already names this person in full ("Robert Allen
            # Hayes" ... "Mr. Hayes" / "Hayes said"): a later bare mention of
            # one of those words is that person, not a product.
            or word.lower() in known_name_parts
        ):
            return score
    return score * PERSON_CONFIDENCE_PENALTY


# Not identifying on its own (HIPAA Safe Harbor permits geography at state
# level and above) - see gazetteers.py.
LOCATION_CONFIDENCE_PENALTY = 0.6


def _adjust_location_confidence(entity_text: str, score: float) -> float:
    if is_non_identifying_place(entity_text):
        return score * LOCATION_CONFIDENCE_PENALTY
    return score


# Same problem as PERSON above, different type: spaCy tags plenty of
# non-dates DATE_TIME ("B.E", "Year", "Semester") at the same flat 0.85 as
# real ones ("11 May 2026"). A real date/time expression, in any format,
# always contains either a digit or a month/weekday name - that's not a
# guess tied to any one document, it's the closed vocabulary of the
# calendar - so anything with neither is almost certainly a mislabel.
DATE_TIME_CONFIDENCE_PENALTY = 0.4
_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_WEEKDAY_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
}


# spaCy's DATE label also covers durations ("7 days", "48 months"),
# frequencies ("monthly", "annual"), relative expressions ("last week",
# "yesterday") and bare years ("2018") - none of which pin down a date tied
# to a person (HIPAA explicitly allows the year alone). Down-weighted so
# they are shown but not pre-selected.
_NUMBER_WORD = (
    r"(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|fifteen|twenty|thirty|few|several|many|couple\s+of)"
)
_DURATION = re.compile(
    rf"^(?:(?:the\s+)?(?:last|next|past|first|previous|coming|over\s+the)\s+)?(?:{_NUMBER_WORD}[\s-]+)?"
    r"(?:days?|weeks?|months?|years?|yrs?|hours?|hrs?|minutes?|mins?|seconds?|secs?|decades?|"
    r"quarters?|fortnights?|semesters?)(?:'s)?(?:\s+(?:ago|old|later|earlier|long))?$",
    re.IGNORECASE,
)
_FREQUENCY_OR_RELATIVE = re.compile(
    r"^(?:daily|weekly|bi-?weekly|monthly|bi-?monthly|quarterly|annual|annually|yearly|"
    r"hourly|nightly|fortnightly|today|tonight|tomorrow|yesterday|now|recently|"
    r"currently|present|this\s+\w+|last\s+\w+|next\s+\w+|the\s+\w+\s+before|"
    r"(?:the\s+)?(?:morning|afternoon|evening|night|weekend|day)s?)$",
    re.IGNORECASE,
)
_BARE_YEAR = re.compile(r"^(?:FY\s?)?(?:19|20)\d{2}(?:\s*[-–/]\s*(?:(?:19|20)?\d{2}|present))?$", re.IGNORECASE)


def _looks_like_a_date(entity_text: str) -> bool:
    stripped = entity_text.strip()
    if _DURATION.match(stripped) or _FREQUENCY_OR_RELATIVE.match(stripped) or _BARE_YEAR.match(stripped):
        return False
    # A bare run of 6+ digits with no separator at all is virtually never
    # how anyone writes a date - it's exactly the shape of a phone number,
    # account number, or other ID instead, and spaCy mislabels those as
    # DATE_TIME often enough that this needs an explicit carve-out: without
    # it, the false positive keeps its full, unpenalized confidence and can
    # win overlap resolution against (and silently swallow) the correctly
    # detected PHONE_NUMBER/ACCOUNT_NUMBER entity for the same span.
    if entity_text.isdigit() and len(entity_text) >= 6:
        return False
    if any(ch.isdigit() for ch in entity_text):
        return True
    words = entity_text.lower().replace(".", " ").split()
    return any(w in _MONTH_NAMES or w in _WEEKDAY_NAMES for w in words)


def _adjust_date_confidence(entity_text: str, score: float) -> float:
    if _looks_like_a_date(entity_text):
        return score
    return score * DATE_TIME_CONFIDENCE_PENALTY


# Presidio's PhoneRecognizer (backed by Google's `phonenumbers` library)
# only ever emits a match after that library has already validated it as a
# structurally correct, dialable number for one of its supported regions
# (which include IN) - but it then gives *every* match the same flat 0.4
# base score regardless, relying entirely on a narrow English context-word
# list ("phone", "mobile", "cell"...) to boost it further. Indian documents
# routinely label numbers as "Ph No:", "Contact:", or with no label at all,
# none of which match that list, so a perfectly valid Indian mobile number
# sits at 0.4 - below AUTO_SELECT_MIN_SCORE - and never gets pre-checked
# for redaction. Since validity is already guaranteed by the recognizer
# itself, floor the score instead of penalizing it; genuine context-word
# matches (already applied by Presidio before this point) can still push it
# higher than the floor.
PHONE_CONFIDENCE_FLOOR = 0.75


def _adjust_phone_confidence(score: float) -> float:
    return max(score, PHONE_CONFIDENCE_FLOOR)


# Same reasoning as the phone floor, for identifiers whose format is
# distinctive enough that a structural match alone is strong evidence.
# Presidio scores UK_NINO 0.5 and relies on context words to lift it, but its
# multi-word context ("national insurance") rarely triggers the boost - so a
# correctly formatted NINO right after "National Insurance Number:" was shown
# unticked. The NINO pattern already enforces the valid prefix letters and
# an A-D suffix, so it is floored to just above the auto-select bar.
VALIDATED_ID_CONFIDENCE_FLOOR = {"UK_NINO": 0.65}


ADDRESS_FAMILY = {"ADDRESS", "LOCATION", "ZIP_CODE", "UK_POSTCODE"}


def _merge_address_family(entities: List[dict], text: str) -> List[dict]:
    """Overlapping address-type spans are pieces of one address, so they are
    merged into a single ADDRESS span rather than competing.

    Without this, a labelled address "B-204, Shanti Nagar Society,
    Navrangpura, Ahmedabad" (0.70) loses overlap resolution to the higher
    scored spaCy sub-span LOCATION "Navrangpura" (0.85), and the rest of the
    address - the flat number and society name - silently leaks. Only
    clusters that contain a structured ADDRESS match are merged; a lone
    LOCATION (a city mention) is left as is.
    """
    family = sorted(
        (e for e in entities if e["type"] in ADDRESS_FAMILY), key=lambda e: e["start"]
    )
    others = [e for e in entities if e["type"] not in ADDRESS_FAMILY]
    merged: List[dict] = []
    cluster: List[dict] = []

    def flush():
        if not cluster:
            return
        if len(cluster) == 1 or not any(e["type"] == "ADDRESS" for e in cluster):
            merged.extend(cluster)
        else:
            start = min(e["start"] for e in cluster)
            end = max(e["end"] for e in cluster)
            best = max(cluster, key=lambda e: e["confidence"])
            merged.append(
                {
                    **best,
                    "type": "ADDRESS",
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "selected": any(e["selected"] for e in cluster),
                }
            )
        cluster.clear()

    for e in family:
        if cluster and e["start"] >= max(c["end"] for c in cluster):
            flush()
        cluster.append(e)
    flush()
    return others + merged


def _resolve_overlaps(entities: List[dict]) -> List[dict]:
    """Presidio's own de-duplication only drops results that share both a
    span *and* an entity type (see EntityRecognizer.remove_duplicates). It
    does not resolve cases like a URL recognizer matching a substring of an
    already-detected EMAIL_ADDRESS, or DATE_TIME and AGE both firing on
    "45-year-old" - so we do a second pass here, keeping the
    highest-confidence entity for any overlapping span."""
    ordered = sorted(
        entities, key=lambda e: (-e["confidence"], e["start"], -(e["end"] - e["start"]))
    )
    kept: List[dict] = []
    for entity in ordered:
        if any(entity["start"] < k["end"] and entity["end"] > k["start"] for k in kept):
            continue
        kept.append(entity)
    return sorted(kept, key=lambda e: e["start"])


def _assert_model_available(spacy_model: str) -> None:
    """Fail fast if the configured spaCy model is not already installed.

    Presidio's NlpEngineProvider silently falls back to `spacy download` when
    it can't find the named model. On a memory-constrained host that is a trap
    rather than a convenience: a typo or a stale platform env var makes the
    container pull ~560MB at startup and then get OOM-killed loading it, which
    surfaces only as a bare "Killed" in the logs with no indication that the
    wrong model was ever requested. The deployed image ships the model it needs,
    so a missing one always means misconfiguration - say so plainly instead.
    """
    if Path(spacy_model).expanduser().is_dir():
        return
    if spacy_model in spacy.util.get_installed_models():
        return
    installed = ", ".join(spacy.util.get_installed_models()) or "none"
    raise RuntimeError(
        f"spaCy model {spacy_model!r} is not installed and is not a directory on "
        f"disk. Installed models: {installed}. Refusing to let Presidio download "
        f"it at runtime - that pulls hundreds of MB into a running container and "
        f"typically ends in an OOM kill. Set SPACY_MODEL to a model present in "
        f"this image (the Docker image ships /app/models/en_core_web_lg_pruned), "
        f"or install the model. Note that a platform env var (Railway/Render "
        f"dashboard variables) overrides the Dockerfile's ENV."
    )


class PiiDetectionEngine:
    """Loads spaCy + Presidio once at startup and reuses it across requests."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        language: str = "en",
        pii_model_dir: Optional[str] = None,
        pii_model_threshold: float = 0.5,
        use_spacy_ner: bool = True,
        use_calibrated_thresholds: bool = True,
    ):
        self.language = language

        # Load the transformer first: onnxruntime briefly holds two copies of
        # the weights while building the session, and doing that before spaCy
        # is resident keeps the process's *peak* memory (what an OOM killer
        # sees) well below steady state + transient.
        gliner = None
        if pii_model_dir:
            from app.detection.gliner_onnx import GlinerOnnxRecognizer

            gliner = GlinerOnnxRecognizer(
                pii_model_dir, threshold=pii_model_threshold, supported_language=language
            )
        _assert_model_available(spacy_model)
        logger.info("Loading spaCy model '%s' for PII detection...", spacy_model)

        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": spacy_model}],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {
                    "PER": "PERSON",
                    "PERSON": "PERSON",
                    "NORP": "NRP",
                    "FAC": "LOCATION",
                    "LOC": "LOCATION",
                    "LOCATION": "LOCATION",
                    "GPE": "LOCATION",
                    "ORG": "ORGANIZATION",
                    "ORGANIZATION": "ORGANIZATION",
                    "DATE": "DATE_TIME",
                    "TIME": "DATE_TIME",
                },
                # Organizations are the noisiest spaCy label (per Presidio's
                # own default config comment) - down-weight rather than drop
                # so they still surface for manual review, at lower confidence.
                "low_confidence_score_multiplier": 0.4,
                "low_score_entity_names": ["ORGANIZATION"],
                "labels_to_ignore": [
                    "CARDINAL", "EVENT", "LANGUAGE", "LAW", "MONEY",
                    "ORDINAL", "PERCENT", "PRODUCT", "QUANTITY", "WORK_OF_ART",
                ],
            },
        }

        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=[language]
        )

        for recognizer_cls in EXTRA_PREDEFINED_RECOGNIZERS:
            self.analyzer.registry.add_recognizer(recognizer_cls(supported_language=language))
        for recognizer in ALL_CUSTOM_RECOGNIZERS:
            self.analyzer.registry.add_recognizer(recognizer)

        self.auto_select_thresholds = (
            _load_calibrated_thresholds() if use_calibrated_thresholds else {}
        )

        self.pii_model_enabled = gliner is not None
        if gliner is not None:
            self.analyzer.registry.add_recognizer(gliner)
        if not use_spacy_ner:
            # spaCy stays loaded for tokens/lemmas (Presidio's context-word
            # boosting needs them) but its statistical NER no longer emits
            # PERSON/LOCATION/ORG - the transformer covers those. Lets a small
            # spaCy pipeline replace the large one on memory-constrained hosts.
            self.analyzer.registry.remove_recognizer("SpacyRecognizer")

        logger.info(
            "PII detection engine ready. Supported entities: %s",
            sorted(self.analyzer.get_supported_entities(language=language)),
        )

    def detect(self, text: str) -> List[dict]:
        """Detect PII entities in `text`, returning the frontend's Entity shape."""
        if not text or not text.strip():
            return []

        # score_threshold intentionally omitted: we want every candidate
        # back from Presidio and apply our own per-type minimum below, in
        # one place, rather than double-gating at two different layers.
        # entities also omitted (see module docstring above) to keep every
        # built-in recognizer - including country-specific ID formats - in play.
        results = self.analyzer.analyze(text=text, language=self.language)

        clipped = []
        for result in results:
            span = self._clip_to_single_line(text, result.start, result.end)
            if span is not None:
                clipped.append((result, span))

        # Words of every full (multi-word) name found anywhere in the document,
        # so a later bare "Hayes" or "Fatima" is recognised as the same person.
        known_name_parts = {
            word.lower().strip(".,")
            for result, (_s, _e, entity_text) in clipped
            if result.entity_type == "PERSON" and _looks_like_a_name(entity_text)
            for word in entity_text.split()
            if len(word.strip(".,")) > 1
        }

        entities = []
        for result, (start, end, entity_text) in clipped:
            score = float(result.score)
            recognizer = (result.recognition_metadata or {}).get("recognizer_name", "")
            if result.entity_type == "PERSON" and recognizer not in CONTEXTUAL_PERSON_RECOGNIZERS:
                score = _adjust_person_confidence(
                    entity_text, score, text, start, end, known_name_parts
                )
            elif result.entity_type == "LOCATION":
                score = _adjust_location_confidence(entity_text, score)
            elif result.entity_type == "DATE_TIME":
                score = _adjust_date_confidence(entity_text, score)
            elif result.entity_type == "PHONE_NUMBER" and recognizer != "LabeledPhoneRecognizer":
                score = _adjust_phone_confidence(score)
            elif result.entity_type in VALIDATED_ID_CONFIDENCE_FLOOR:
                score = max(score, VALIDATED_ID_CONFIDENCE_FLOOR[result.entity_type])

            min_score = MIN_SCORE_BY_TYPE.get(result.entity_type, DEFAULT_MIN_SCORE)
            if score < min_score:
                continue

            entities.append(
                {
                    "text": entity_text,
                    "type": result.entity_type,
                    "start": start,
                    "end": end,
                    "confidence": round(score, 4),
                    "source": "presidio",
                    "selected": score
                    >= self.auto_select_thresholds.get(result.entity_type, AUTO_SELECT_MIN_SCORE),
                }
            )

        return _resolve_overlaps(_merge_address_family(entities, text))

    @staticmethod
    def _clip_to_single_line(text: str, start: int, end: int):
        """spaCy occasionally merges an entity with the start of the next
        line (e.g. a name header immediately followed by "Email:" on a
        resume becomes one PERSON span). PII fields are effectively always
        single-line, so truncate at the first newline and drop anything
        that becomes too short to be meaningful."""
        raw = text[start:end]
        newline_idx = raw.find("\n")
        if newline_idx != -1:
            raw = raw[:newline_idx]

        stripped = raw.strip()
        if len(stripped) < 2:
            return None

        leading_ws = len(raw) - len(raw.lstrip())
        clipped_start = start + leading_ws
        return clipped_start, clipped_start + len(stripped), stripped
