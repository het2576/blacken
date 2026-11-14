"""Unit tests for the evaluation metrics themselves - a metric bug would
silently inflate or deflate every number in the report."""

from evals.datasets import parse_markup
from evals.metrics import aggregate, prf, score_document


def test_parse_markup_offsets():
    text, spans = parse_markup("Call [[PERSON|Asha Rao]] at [[PHONE|98450 31276]].")
    assert text == "Call Asha Rao at 98450 31276."
    assert [(text[s:e], c) for s, e, c in spans] == [("Asha Rao", "PERSON"), ("98450 31276", "PHONE")]


def test_prf_and_f2():
    m = prf(tp=8, fp=2, fn=8)
    assert m["precision"] == 0.8 and m["recall"] == 0.5
    assert abs(m["f1"] - 0.6153846) < 1e-6
    assert abs(m["f2"] - (5 * 0.8 * 0.5 / (4 * 0.8 + 0.5))) < 1e-9


def test_perfect_prediction_scores_one():
    text = "John Smith lives at 42 Wallaby Way."
    gold = [(0, 10, "PERSON"), (20, 34, "ADDRESS")]
    r = aggregate([score_document(text, gold, gold)])
    assert r["token"]["f1"] == 1.0 and r["overlap"]["f1"] == 1.0 and r["strict"]["f1"] == 1.0


def test_partial_name_counts_as_overlap_but_not_full_coverage():
    text = "John Smith called."
    r = aggregate([score_document(text, [(0, 10, "PERSON")], [(5, 10, "PERSON")])])
    assert r["overlap"]["recall"] == 1.0
    assert r["full_coverage_recall"] == 0.0
    assert r["token"]["recall"] == 0.5
    assert r["strict"]["recall"] == 0.0


def test_ignore_and_org_regions_are_neutral():
    text = "user gthompson88 at Acme Corp"
    gold = [(5, 16, "IGNORE"), (20, 29, "ORGANIZATION")]
    pred = [(5, 16, "PERSON"), (20, 29, "ORGANIZATION")]
    r = aggregate([score_document(text, gold, pred)])
    assert r["predicted_entities"] == 0 and r["token"]["fp"] == 0


def test_false_alarm_on_negative_document():
    r = aggregate([score_document("Docker and Kubernetes.", [], [(0, 6, "PERSON")])])
    assert r["negative_false_alarm_rate"] == 1.0
    assert r["token"]["fp"] == 1


def test_trimming_punctuation_for_strict_match():
    text = "Email: a@b.com."
    r = aggregate([score_document(text, [(7, 14, "EMAIL")], [(7, 15, "EMAIL")])])
    assert r["strict"]["f1"] == 1.0
