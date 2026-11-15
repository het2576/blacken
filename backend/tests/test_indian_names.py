"""Regression tests for real failures on Indian-format documents and resumes
(reported from real uploads, 2026-11-15). See evals/data/indian_gold.txt for
the full measured suite."""

from app.detection.indian_names import clean_person_span, is_tech_or_code


def _persons(engine, text):
    return {e["text"]: e for e in engine.detect(text) if e["type"] == "PERSON"}


def test_all_caps_form_names_detected(engine):
    text = "NAME: RAHUL KUMAR SHARMA\nFATHER'S NAME: SANJAY SHARMA\nMR. ARJUN REDDY PATEL"
    found = _persons(engine, text)
    for name in ("RAHUL KUMAR SHARMA", "SANJAY SHARMA", "ARJUN REDDY PATEL"):
        assert name in found and found[name]["selected"], (name, found)


def test_relation_markers_split_and_detect_both_people(engine):
    text = "Applicant Suresh Babu Naidu, S/o Ramana Naidu, resident of Hyderabad."
    found = _persons(engine, text)
    assert "Suresh Babu Naidu" in found and found["Suresh Babu Naidu"]["selected"]
    assert "Ramana Naidu" in found and found["Ramana Naidu"]["selected"]


def test_ocr_run_on_relation_line(engine):
    text = "Name RAJU SHARMA Son of Mohan Sharma Date of Birth 15 08 1985"
    found = _persons(engine, text)
    assert {"RAJU SHARMA", "Mohan Sharma"} <= set(found), found
    assert all(found[n]["selected"] for n in ("RAJU SHARMA", "Mohan Sharma"))


def test_short_indian_first_names_are_people_not_places(engine):
    entities = engine.detect("Please welcome Disha and Kavya. Sudhanshu followed you.")
    by_text = {e["text"]: e for e in entities}
    for name in ("Disha", "Kavya", "Sudhanshu"):
        assert by_text[name]["type"] == "PERSON", by_text
        assert by_text[name]["selected"]


def test_resume_header_name_detected_and_tech_stack_left_alone(engine):
    text = (
        "Jay Ladhaniya\n+91 98765 43210 | someone@example.com\n"
        "Blacken | FastAPI, React, TypeScript, PyMuPDF, PostgreSQL, Next.js, Docker"
    )
    entities = engine.detect(text)
    assert any(e["type"] == "PERSON" and e["text"] == "Jay Ladhaniya" and e["selected"] for e in entities)
    flagged = {e["text"] for e in entities}
    assert not flagged & {"React", "TypeScript", "PyMuPDF", "PostgreSQL", "Next.js", "Docker"}


def test_institutions_named_after_people_are_not_people(engine):
    text = "The cargo reached Jawaharlal Nehru Port and then Mahatma Gandhi Road near Shiv Nadar University."
    found = _persons(engine, text)
    assert not any(e["selected"] for e in found.values()), found


def test_sloppy_model_span_is_trimmed_to_the_name():
    text = "Passenger 1: Ashwath Hegdekatte / Male / 29"
    start = text.index("Ashwath")
    pieces = [text[s:e] for s, e in clean_person_span(text, start, len(text))]
    assert pieces == ["Ashwath Hegdekatte"]


def test_tech_detection_keeps_real_name_prefixes():
    assert is_tech_or_code("PyMuPDF") and is_tech_or_code("Next.js") and is_tech_or_code("C++")
    assert not is_tech_or_code("McDonald") and not is_tech_or_code("DeSouza") and not is_tech_or_code("Priya")
