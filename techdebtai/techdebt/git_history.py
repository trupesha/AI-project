"""Mine a git repository: per-commit, per-file change history + bug-fix labelling."""
import re
import subprocess
from datetime import datetime, timezone

import pandas as pd

# Heuristic: commit messages that look like bug fixes (replace with issue-tracker links if you have them)
BUG_RE = re.compile(r"\b(fix(es|ed)?|bug(s)?|defect|hotfix|patch|crash|regression)\b", re.I)


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, check=True, encoding="utf-8", errors="replace",
    ).stdout


def load_history(repo: str) -> pd.DataFrame:
    """One row per (commit, file): commit, author, ts, is_bug, path, added, deleted."""
    fmt = "@@%H\x1f%an\x1f%at\x1f%s"
    out = git(repo, "log", "--no-merges", "--no-renames", "--numstat", f"--format={fmt}")
    rows, cur = [], None
    for line in out.splitlines():
        if line.startswith("@@"):
            h, a, t, s = line[2:].split("\x1f", 3)
            cur = (h, a, int(t), bool(BUG_RE.search(s)))
        elif cur and line.strip():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0] != "-":  # skip binary files
                rows.append((*cur, parts[2], int(parts[0]), int(parts[1])))
    cols = ["commit", "author", "ts", "is_bug", "path", "added", "deleted"]
    return pd.DataFrame(rows, columns=cols)


def rev_at(repo: str, ts: int) -> str:
    """Last commit on HEAD's history at or before unix time `ts`."""
    when = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")
    return git(repo, "rev-list", "-1", f"--before={when}", "HEAD").strip()
