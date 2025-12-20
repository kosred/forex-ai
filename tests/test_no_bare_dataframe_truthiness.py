import pathlib
import re

PATTERNS = [
    re.compile(r"\\bif\\s+df\\s*:", re.IGNORECASE),
    re.compile(r"\\bif\\s+not\\s+df\\s*:", re.IGNORECASE),
    re.compile(r"\\bwhile\\s+df\\s*:", re.IGNORECASE),
    re.compile(r"\\bwhile\\s+not\\s+df\\s*:", re.IGNORECASE),
]


def test_no_bare_dataframe_truthiness():
    """
    Guardrail: avoid ambiguous truthiness on pandas DataFrames in new code.
    (Use .empty/.any()/.all() instead.)
    """
    root = pathlib.Path("src")
    bad_hits = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                bad_hits.append((path, match.group(0).strip()))
    assert not bad_hits, f"Bare DataFrame truthiness detected: {bad_hits}"
