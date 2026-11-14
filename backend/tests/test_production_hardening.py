"""Regression tests for defects found by the evaluation harness (evals/).
Each test pins one measured failure so it cannot silently come back."""

import io
import zipfile

import pymupdf
import pytest
from docx import Document

from app.redaction.matching import find_matches


# --------------------------------------------------------------- matching
def test_matching_respects_token_boundaries():
    # Old substring search redacted the "52" inside "1,52,400" and "352".
    text = "Age 52. Paid 1,52,400 in Ward 352. Age again: 52"
    spans = find_matches(text, ["52"])
    assert [text[s:e] for s, e in spans] == ["52", "52"]


def test_matching_tolerates_whitespace_differences():
    text = "Ahmedabad\t 380009"
    assert find_matches(text, ["Ahmedabad 380009"]) == [(0, len(text))]


def test_matching_is_case_insensitive_and_merges_overlaps():
    text = "JOHN SMITH and john smith"
    assert len(find_matches(text, ["John Smith", "Smith"])) == 2


# -------------------------------------------------------------- detection
@pytest.mark.parametrize(
    "text,value,etype",
    [
        ("Aadhaar No: 3588 1399 8697", "3588 1399 8697", "IN_AADHAAR"),
        ("PAN: AFZPM7190K", "AFZPM7190K", "IN_PAN"),
        ("National Insurance Number: JG 10 37 72 D", "JG 10 37 72 D", "UK_NINO"),
        ("Employee Code: LT482019", "LT482019", "EMPLOYEE_ID"),
        ("Routing: 121000358", "121000358", "ABA_ROUTING_NUMBER"),
        ("Payment via UPI: arjunmehta@okicici", "arjunmehta@okicici", "UPI_ID"),
    ],
)
def test_identifiers_previously_missed_are_detected_and_selected(engine, text, value, etype):
    by_text = {e["text"]: e for e in engine.detect(text)}
    assert value in by_text, engine.detect(text)
    assert by_text[value]["type"] == etype
    assert by_text[value]["selected"] is True


def test_labeled_address_is_one_span_not_dropped_for_subspan(engine):
    """A sub-span LOCATION used to win overlap resolution and the rest of the
    address (flat number, society) was silently dropped."""
    text = "Address: B-204, Shanti Nagar Society, Navrangpura, Ahmedabad - 380009"
    addresses = [e for e in engine.detect(text) if e["type"] == "ADDRESS"]
    assert addresses and "B-204" in addresses[0]["text"]
    assert "Navrangpura" in addresses[0]["text"]
    assert addresses[0]["selected"] is True


def test_single_names_in_clear_person_context_are_selected(engine):
    text = "Robert Allen Hayes was admitted. Mr. Hayes improved.\nThanks Rahul.\nCheers,\nMegan"
    selected = {e["text"] for e in engine.detect(text) if e["type"] == "PERSON" and e["selected"]}
    assert {"Hayes", "Rahul", "Megan"} <= selected


def test_countries_and_durations_are_not_auto_selected(engine):
    text = "Resident of United Kingdom. Take for 7 days, review monthly. Income annual."
    for e in engine.detect(text):
        if e["text"] in {"United Kingdom", "7 days", "monthly", "annual"}:
            assert e["selected"] is False, e


# ---------------------------------------------------------------- helpers
def _docx_with_table_and_header() -> bytes:
    doc = Document()
    doc.core_properties.author = "Rosalind Featherstone"
    doc.sections[0].header.paragraphs[0].text = "Patient: Robert Johnson"
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "SSN:", "234-56-7890"
    table.cell(1, 0).text, table.cell(1, 1).text = "Email:", "robert.johnson@example.com"
    doc.add_paragraph("Routine checkup with no other findings.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _all_docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return " ".join(z.read(n).decode("utf-8", "ignore") for n in z.namelist() if n.endswith(".xml"))


def _upload_redact_download(client, name, data):
    up = client.post("/api/upload", files={"file": (name, data, "application/octet-stream")})
    assert up.status_code == 200, up.text
    doc_id = up.json()["document_id"]
    rr = client.post("/api/redact", json={"document_id": doc_id})
    assert rr.status_code == 200, rr.text
    return up.json(), client.get(f"/api/download/{doc_id}").content


# ------------------------------------------------------------- redaction
def test_docx_tables_headers_and_metadata_are_redacted(api_client):
    upload, out = _upload_redact_download(api_client, "form.docx", _docx_with_table_and_header())
    detected = {e["text"] for e in upload["entities"]}
    assert {"234-56-7890", "robert.johnson@example.com", "Robert Johnson"} <= detected

    xml = _all_docx_text(out)
    for leaked in ("234-56-7890", "robert.johnson@example.com", "Robert Johnson", "Rosalind Featherstone"):
        assert leaked not in xml, f"{leaked!r} survived in output DOCX"
    assert "Routine checkup" in xml


def test_pdf_metadata_scrubbed_and_neighbour_lines_untouched(api_client):
    doc = pymupdf.open()
    page = doc.new_page()
    lines = ["Patient Name: Robert Johnson", "SSN: 234-56-7890", "Routine checkup completed today."]
    for i, line in enumerate(lines):
        page.insert_text((50, 72 + 12 * i), line, fontsize=10)  # tight leading
    doc.set_metadata({"author": "Rosalind Featherstone"})
    buf = io.BytesIO()
    doc.save(buf)

    _, out = _upload_redact_download(api_client, "tight.pdf", buf.getvalue())
    red = pymupdf.open(stream=out, filetype="pdf")
    text = red[0].get_text()
    assert "234-56-7890" not in text and "Robert Johnson" not in text
    # Words on the lines adjacent to a redaction must survive.
    assert "Routine checkup completed today." in text
    assert "Patient Name:" in text
    assert "Rosalind" not in str(red.metadata)


def test_redact_with_nothing_selected_returns_clean_copy_not_500(api_client):
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), "Plain text with no personal data in it at all.")
    buf = io.BytesIO()
    doc.save(buf)
    up = api_client.post("/api/upload", files={"file": ("none.pdf", buf.getvalue(), "application/pdf")})
    doc_id = up.json()["document_id"]
    rr = api_client.post(
        "/api/redact",
        json={"document_id": doc_id, "entities": [{"text": "x", "type": "PERSON", "start": 0, "end": 1, "selected": False}]},
    )
    assert rr.status_code == 200, rr.text
    assert api_client.get(f"/api/download/{doc_id}").status_code == 200


def test_txt_files_can_be_redacted(api_client):
    _, out = _upload_redact_download(
        api_client, "note.txt", b"Contact john.smith@example.com, SSN 234-56-7890. Thanks."
    )
    text = out.decode("utf-8")
    assert "john.smith@example.com" not in text and "234-56-7890" not in text
    assert "Thanks." in text


# --------------------------------------------------------------- security
def test_missing_owner_id_is_rejected(api_client):
    r = api_client.post(
        "/api/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        headers={"X-Owner-Id": ""},
    )
    assert r.status_code == 400
    assert api_client.get("/api/documents", headers={"X-Owner-Id": ""}).status_code == 400


def test_owners_cannot_see_each_others_documents(api_client):
    up = api_client.post("/api/upload", files={"file": ("a.txt", b"SSN 234-56-7890", "text/plain")})
    doc_id = up.json()["document_id"]
    other = {"X-Owner-Id": "someone-else-000000000000"}
    assert api_client.get("/api/documents", headers=other).json()["documents"] == []
    assert api_client.get(f"/api/download/{doc_id}?original=true", headers=other).status_code == 404
    assert api_client.delete(f"/api/document/{doc_id}", headers=other).status_code == 404


def test_runtime_config_is_not_writable_anonymously(api_client):
    r = api_client.patch("/api/config", json={"retention_hours": 720})
    assert r.status_code == 403
    assert api_client.get("/api/config").json()["retention_hours"] != 720
