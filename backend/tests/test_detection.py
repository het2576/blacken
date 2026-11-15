"""Unit tests for the Presidio-based detection engine.

The `engine` fixture (tests/conftest.py) loads the real spaCy model once
per test session - these are deliberately not mocked, since the whole
point is to catch regressions in real detection accuracy, which is
exactly what silently broke in the previous hand-rolled implementation.
"""


def test_detects_common_pii_types(engine):
    text = (
        "Contact John Smith at john.smith@example.com or (555) 123-4567. "
        "SSN: 234-56-7890. Card on file: 4532015112830366."
    )
    entities = engine.detect(text)
    found = {(e["type"], e["text"]) for e in entities}

    assert ("PERSON", "John Smith") in found
    assert ("EMAIL_ADDRESS", "john.smith@example.com") in found
    assert ("CREDIT_CARD", "4532015112830366") in found
    assert any(t == "PHONE_NUMBER" for t, _ in found)
    assert any(t == "US_SSN" for t, _ in found)


def test_no_false_positives_on_generic_medical_text(engine):
    """This is the exact failure mode of the previous engine: names/terms
    tied to one sample document were hardcoded into a blocklist instead of
    the detector generalizing. Plain clinical language with no PII in it
    should produce zero detections."""
    text = (
        "The patient was diagnosed with diabetes and hypertension. "
        "Treatment includes medication and follow-up care. "
        "Blood pressure was 120/80 mmHg and remains well controlled."
    )
    assert engine.detect(text) == []


def test_custom_recognizers_capture_identifier_not_label(engine):
    text = "Medical record: MRN-882910. Patient ID: 55210. Account Number: 774411."
    by_type = {e["type"]: e["text"] for e in engine.detect(text)}

    assert by_type.get("MEDICAL_RECORD_NUMBER") == "882910"
    assert by_type.get("PATIENT_ID") == "55210"
    assert by_type.get("ACCOUNT_NUMBER") == "774411"


def test_address_and_zip_recognizers(engine):
    text = "She lives at 42 Wallaby Way. Zip Code: 90210."
    by_type = {e["type"]: e["text"] for e in engine.detect(text)}

    assert by_type.get("ADDRESS") == "42 Wallaby Way"
    assert by_type.get("ZIP_CODE") == "90210"


def test_entities_never_overlap(engine):
    text = (
        "John Smith, a 45-year-old, emailed john.smith@example.com "
        "on 2024-01-15 regarding his SSN 234-56-7890."
    )
    entities = sorted(engine.detect(text), key=lambda e: e["start"])
    for a, b in zip(entities, entities[1:]):
        assert a["end"] <= b["start"], f"Overlapping entities: {a} and {b}"


def test_entities_never_span_a_newline(engine):
    """Regression test: spaCy sometimes merges a name header with the
    start of the next line (e.g. a resume's "Jane Doe\\nEmail: ...")."""
    text = "Jane Doe\nEmail: jane.doe@example.com\nPhone: 415-555-2671"
    for entity in engine.detect(text):
        assert "\n" not in entity["text"]


def test_empty_text_returns_no_entities(engine):
    assert engine.detect("") == []
    assert engine.detect("   \n\t  ") == []


def test_real_person_stays_selected_by_default(engine):
    text = "Contact John Smith at john.smith@example.com regarding the invoice."
    by_text = {e["text"]: e for e in engine.detect(text)}
    assert by_text["John Smith"]["selected"] is True
    assert by_text["john.smith@example.com"]["selected"] is True


def test_single_word_tech_terms_not_auto_selected_as_person(engine):
    """Regression test for a real report: a resume's skills list ("Docker,
    Postman, Claude, Streamlit, ...") got tagged PERSON by spaCy's NER and
    every single one defaulted to selected=True, so a legitimate document
    review looked like "74 of 74 selected" full of tool/brand names. spaCy
    gives every PERSON hit the same flat confidence regardless of how
    plausible it is, so single bare capitalized words - structurally much
    more likely to be a brand/tool than a bare first name in a document -
    get down-weighted and shouldn't be pre-checked for redaction."""
    text = "Skills: Docker, Postman, Streamlit, and NextAuth."
    entities = engine.detect(text)
    # Known software words are now dropped outright (indian_names.TECH_TERMS /
    # is_tech_or_code), so they don't even clutter the review list.
    name_like = [e for e in entities if e["type"] in {"PERSON", "LOCATION", "ORGANIZATION", "NRP"}]
    assert not any(e["selected"] for e in name_like), name_like
    assert not {"Docker", "Postman", "Streamlit", "NextAuth"} & {e["text"] for e in name_like}


def test_multiword_capitalized_name_keeps_full_confidence(engine):
    text = "Please welcome our new hire, Maria Gonzalez, to the team."
    by_text = {e["text"]: e for e in engine.detect(text)}
    assert by_text["Maria Gonzalez"]["confidence"] >= 0.8
    assert by_text["Maria Gonzalez"]["selected"] is True
