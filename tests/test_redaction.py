from redaction import (LineRedactor, RedactingFilter, detect_secret_kinds, redact, redact_obj,
                       register_secret)
from tests.fakes import FAKE_AWS_KEY, FAKE_GH_TOKEN
import logging


def test_aws_key_redacted():
    out = redact(f'KEY = "{FAKE_AWS_KEY}"')
    assert FAKE_AWS_KEY not in out and "[REDACTED:aws_access_key_id]" in out


def test_github_token_redacted():
    assert FAKE_GH_TOKEN not in redact(f"token: {FAKE_GH_TOKEN}")


def test_generic_assignment_redacted_but_placeholders_kept():
    assert "hunter22x" not in redact('password = "hunter22x"')
    assert "your_api_key_here" in redact("api_key = your_api_key_here")
    assert "${DB_PASSWORD}" in redact("password: ${DB_PASSWORD}")
    assert 'os.environ["SECRET_KEY"]' in redact('secret = os.environ["SECRET_KEY"]')
    assert "${{ secrets.TOKEN }}" in redact("token: ${{ secrets.TOKEN }}")


def test_json_quoted_key_redacted():
    assert "s3cr3tValue9" not in redact('{"client_secret": "s3cr3tValue9"}')


def test_numeric_config_not_flagged():
    assert detect_secret_kinds("max_tokens = 100000") == []


def test_url_password_redacted_user_kept():
    out = redact("postgres://admin:Sup3rS3cret@db.example.org/app")
    assert "Sup3rS3cret" not in out and "admin" in out


def test_private_key_block_hidden_line_by_line():
    r = LineRedactor()
    lines = ["-----BEGIN RSA PRIVATE KEY-----", "MIIEowIBAAKCAQEA1234567890", "abcdef", "-----END RSA PRIVATE KEY-----", "x = 1"]
    shown = [r.redact_line(ln)[0] for ln in lines]
    assert all("MIIE" not in s and "abcdef" not in s for s in shown)
    assert shown[-1] == "x = 1"


def test_registered_secret_is_scrubbed_everywhere():
    register_secret("my-very-private-api-key-123")
    assert "my-very-private" not in redact("calling with my-very-private-api-key-123 now")


def test_detect_reports_kinds_not_values():
    kinds = detect_secret_kinds(f'k = "{FAKE_AWS_KEY}"')
    assert kinds == ["aws_access_key_id"]
    assert detect_secret_kinds("AKIAIOSFODNN7EXAMPLE") == []  # documented AWS example key


def test_redact_obj_recurses():
    obj = {"a": [f"x {FAKE_AWS_KEY}"], "b": {"c": 3}}
    assert FAKE_AWS_KEY not in str(redact_obj(obj))


def test_logging_filter_scrubs(caplog):
    logger = logging.getLogger("devops_agent.test_filter")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger="devops_agent.test_filter"):
        logger.info("found %s", FAKE_AWS_KEY)
    # caplog handler sees the record after logger-level filters
    assert FAKE_AWS_KEY not in caplog.text
