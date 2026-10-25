from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.fakes import FAKE_AWS_KEY


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=root, check=True, capture_output=True,
    )


def make_repo(root: Path, files: dict[str, str | bytes], commit_msgs: tuple[str, ...] = ("initial commit",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", commit_msgs[0])
    for i, msg in enumerate(commit_msgs[1:], 1):
        (root / f"change{i}.txt").write_text(str(i))
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", msg)
    return root


BAD_WORKFLOW = """\
name: CI
on: pull_request_target
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - uses: some-org/some-action@main
      - run: echo "Title is ${{ github.event.pull_request.title }}"
      - run: pytest || true
"""

GOOD_WORKFLOW = """\
name: CI
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest
"""


@pytest.fixture()
def bad_repo(tmp_path: Path) -> Path:
    return make_repo(
        tmp_path / "bad",
        {
            "app.py": "import os\n\n\ndef main():\n    return 1\n",
            "config.py": f'AWS_ACCESS_KEY_ID = "{FAKE_AWS_KEY}"\nDEBUG = True\nDATA_DIR = "/home/alice/data"\n',
            ".github/workflows/ci.yml": BAD_WORKFLOW,
            "__pycache__/app.cpython-311.pyc": b"\x00\x01\x02binary",
            ".DS_Store": b"\x00\x00junk",
        },
        commit_msgs=("initial commit", "add stuff", "fix things"),
    )


@pytest.fixture()
def good_repo(tmp_path: Path) -> Path:
    return make_repo(
        tmp_path / "good",
        {
            "README.md": "# Demo\n\nInstall with pip.\n",
            ".gitignore": "__pycache__/\n.env\n",
            "pyproject.toml": '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["requests>=2,<3"]\n',
            "requirements.txt": "requests>=2,<3\n",
            "src/demo.py": "def add(a, b):\n    return a + b\n",
            "tests/test_demo.py": "from src.demo import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            ".github/workflows/ci.yml": GOOD_WORKFLOW,
        },
    )
