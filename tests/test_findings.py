import json

import pytest

from findings import (FIELD_ORDER, FindingStore, build_report, validate_report, write_report)
from tests.fakes import FAKE_AWS_KEY
from tools import RepoTools


@pytest.fixture()
def store(bad_repo):
    t = RepoTools(bad_repo)
    t.execute("list_files", {})
    return FindingStore(t)


def valid(**over):
    base = {"category": "security", "severity": "critical", "file": None, "line": None,
            "description": "Hardcoded AWS access key", "evidence": "config.py assigns an AWS access key id literal to a variable",
            "recommendation": "Load credentials from environment or a secrets manager and rotate the key"}
    base.update(over)
    return base


def test_accepts_valid_repo_wide_finding(store):
    ok, msg = store.add(valid())
    assert ok and "SEC-001" in msg


def test_requires_exploration_first(bad_repo):
    s = FindingStore(RepoTools(bad_repo))
    ok, msg = s.add(valid())
    assert not ok and "explore" in msg


def test_rejects_nonexistent_file(store):
    ok, msg = store.add(valid(file="nope.py"))
    assert not ok and "does not exist" in msg


def test_line_must_have_been_seen(store):
    ok, msg = store.add(valid(file="config.py", line=1))
    assert not ok and "never shown" in msg
    store.tools.execute("read_file", {"path": "config.py"})
    ok, _ = store.add(valid(file="config.py", line=1))
    assert ok


def test_line_requires_file_and_positive_int(store):
    assert not store.add(valid(line=3))[0]
    store.tools.execute("read_file", {"path": "config.py"})
    assert not store.add(valid(file="config.py", line=0))[0]
    assert not store.add(valid(file="config.py", line="x"))[0]


def test_string_nulls_normalised(store):
    ok, _ = store.add(valid(file="null", line="none"))
    assert ok
    assert store.finalize()[0]["file"] is None and store.finalize()[0]["line"] is None


def test_missing_paths_contradiction_rejected(store):
    ok, msg = store.add(valid(category="repository_hygiene", severity="low", description="Missing app module",
                              evidence="No application module was found in the repository root", missing_paths=["app.py"]))
    assert not ok and "exists" in msg
    ok, _ = store.add(valid(category="repository_hygiene", severity="low", description="No README present",
                            evidence="find_files('README*') returned no matches", missing_paths=["README*"]))
    assert ok


def test_duplicates_rejected(store):
    store.tools.execute("read_file", {"path": "config.py"})
    assert store.add(valid(file="config.py", line=1))[0]
    ok, msg = store.add(valid(file="config.py", line=1, description="Different wording entirely here"))
    assert not ok and "duplicate" in msg
    ok, msg = store.add(valid(description="Hardcoded AWS access key!"))  # file null: near-identical text
    assert ok  # different file scope (null vs config.py) is not a duplicate
    ok, msg = store.add(valid(description="Hardcoded AWS access key!!"))
    assert not ok and "duplicate" in msg


def test_secrets_in_text_are_redacted(store):
    ok, _ = store.add(valid(evidence=f"The value {FAKE_AWS_KEY} is assigned in config.py directly"))
    assert ok
    assert FAKE_AWS_KEY not in json.dumps(store.finalize())


def test_category_aliases_and_validation(store):
    assert store.add(valid(category="CI"))[0]
    assert store.add(valid(category="GitHub Actions", description="Another CI thing worth noting"))[0]
    assert not store.add(valid(category="mlops"))[0]
    assert not store.add(valid(severity="catastrophic", description="Something else entirely wrong"))[0]


def test_short_evidence_rejected(store):
    assert not store.add(valid(evidence="bad"))[0]
    assert not store.add(valid(recommendation=""))[0]


def test_finalize_sorts_by_severity_and_renumbers(store):
    store.add(valid(severity="low", category="repository_hygiene", description="Minor hygiene issue one"))
    store.add(valid(severity="critical", description="Critical credential problem"))
    store.add(valid(severity="high", category="repository_hygiene", description="Hygiene issue two significant"))
    out = store.finalize()
    assert [v["severity"] for v in out] == ["critical", "high", "low"]
    assert [v["rule_id"] for v in out] == ["SEC-001", "HYG-001", "HYG-002"]
    assert all(list(v) == list(FIELD_ORDER) for v in out)


def test_withdraw(store):
    store.add(valid())
    assert store.withdraw("SEC-001", "fp")[0]
    assert len(store) == 0
    assert not store.withdraw("SEC-009", "x")[0]


def test_validate_report_contract():
    assert validate_report({"repository": "https://github.com/a/b", "violations": []}) == []
    assert validate_report({"repository": "x", "violations": [], "extra": 1})
    assert validate_report({"repository": "x", "violations": [{"rule_id": "SEC-001"}]})
    bad = {k: "x" for k in FIELD_ORDER}
    bad.update(rule_id="SEC-001", severity="urgent", file=None, line=None)
    assert any("severity" in e for e in validate_report({"repository": "x", "violations": [bad]}))


def test_write_report_atomic_and_valid(tmp_path):
    out = tmp_path / "sub" / "report.json"
    write_report(out, build_report("https://github.com/a/b", []))
    assert json.loads(out.read_text()) == {"repository": "https://github.com/a/b", "violations": []}
    assert [p.name for p in out.parent.iterdir()] == ["report.json"]  # no temp files left behind
    with pytest.raises(ValueError):
        write_report(out, {"repository": "x"})
    assert json.loads(out.read_text())["violations"] == []  # untouched by the failed write
