"""Named case subsets. `FAST_CASES` is what `make eval-fast` and the every-PR CI job run.

It keeps every category and both splits, every write/permission case, the PII cases that
touch each identifier type, the main failure modes, a multi-hop and stale-record slice, and
a dev/held-out slice of each attack and benign surface, at about a third of the suite.
`make eval` always runs everything.
"""

from __future__ import annotations

FAST_CASES: tuple[str, ...] = (
    # answers and citation resolution
    "norm-001", "norm-003", "norm-004", "norm-007", "norm-016",
    "edge-001", "edge-004", "edge-006",
    "mem-001", "mem-003",
    # input attacks (dev, then held-out)
    "au-d01", "au-d04", "au-d05", "au-d07", "au-d09",
    "au-h02", "au-h05", "au-h07", "au-h09",
    # retrieved-content attacks
    "ar-d01", "ar-d03", "ar-d05", "ar-d07", "ar-d11",
    "ar-h01", "ar-h03", "ar-h06",
    # benign lookalikes (input and retrieved)
    "bl-d01", "bl-d03", "bl-d06", "bl-h01", "bl-h03",
    "bl-d10", "bl-d13", "bl-h07", "bl-h09",
    # permissions (all)
    "perm-001", "perm-002", "perm-003", "perm-004", "perm-005", "perm-006", "perm-007",
    # PII
    "pii-001", "pii-003", "pii-005",
    # tool failures
    "tf-001", "tf-003", "tf-004", "tf-005", "tf-006", "tf-007",
)  # fmt: skip
