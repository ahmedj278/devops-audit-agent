import json
from pathlib import Path

from findings import validate_report
from redaction import detect_secret_kinds

SAMPLE = Path(__file__).resolve().parent.parent / "sample_report.json"


def test_sample_report_is_schema_valid():
    report = json.loads(SAMPLE.read_text())
    assert validate_report(report) == []
    assert report["violations"], "the sample should demonstrate real findings"


def test_sample_report_contains_no_secrets():
    for line in SAMPLE.read_text().splitlines():
        assert detect_secret_kinds(line) == [], line


def test_sample_findings_have_evidence_and_recommendation():
    for v in json.loads(SAMPLE.read_text())["violations"]:
        assert len(v["evidence"]) >= 15 and len(v["recommendation"]) >= 10
