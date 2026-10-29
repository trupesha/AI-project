"""Feature engineering: code metrics + change history + bug history -> one row per file."""
import numpy as np
import pandas as pd

from .git_history import rev_at
from .static_metrics import analyze_snapshot

DAY = 86400
STATIC = ["nloc", "n_functions", "avg_ccn", "max_ccn", "avg_params", "max_func_nloc",
          "todo_count", "fan_in", "fan_out"]
HISTORY = ["n_commits", "n_authors", "top_author_share", "churn", "churn_per_commit",
           "commits_90d", "bug_fixes", "bug_fix_ratio", "age_days", "days_since_change"]
FEATURES = STATIC + HISTORY


def history_features(hist: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    h = hist[hist.ts < cutoff]
    g = h.groupby("path")
    out = pd.DataFrame({
        "n_commits": g.commit.nunique(),
        "n_authors": g.author.nunique(),
        "churn": g.added.sum() + g.deleted.sum(),
        "first": g.ts.min(),
        "last": g.ts.max(),
    })
    out["bug_fixes"] = h[h.is_bug].groupby("path").commit.nunique()
    out["commits_90d"] = h[h.ts >= cutoff - 90 * DAY].groupby("path").commit.nunique()
    by_author = h.groupby(["path", "author"]).commit.nunique().rename("c").reset_index()
    out["top_author_share"] = by_author.groupby("path").c.max() / by_author.groupby("path").c.sum()
    out = out.fillna(0)
    out["churn_per_commit"] = out.churn / out.n_commits.clip(lower=1)
    out["bug_fix_ratio"] = out.bug_fixes / out.n_commits.clip(lower=1)
    out["age_days"] = (cutoff - out["first"]) / DAY
    out["days_since_change"] = (cutoff - out["last"]) / DAY
    return out.drop(columns=["first", "last"])


def future_bug_label(hist: pd.DataFrame, cutoff: int, horizon_s: int) -> pd.Series:
    """1 if the file received a bug-fix commit in (cutoff, cutoff+horizon]."""
    fut = hist[(hist.ts >= cutoff) & (hist.ts < cutoff + horizon_s) & hist.is_bug]
    return pd.Series(1, index=fut.path.unique(), name="label")


def build_snapshot(repo: str, hist: pd.DataFrame, cutoff: int, rev: str = None) -> pd.DataFrame:
    rev = rev or rev_at(repo, cutoff)
    static = analyze_snapshot(repo, rev)
    if static.empty:
        return static
    df = static.join(history_features(hist, cutoff), how="left").fillna(0)
    return df[FEATURES]


def heuristic_debt_score(df: pd.DataFrame) -> pd.Series:
    """Unsupervised 0-100 'current debt' score: weighted percentile ranks of risk signals."""
    w = {"avg_ccn": .12, "max_ccn": .15, "max_func_nloc": .10, "nloc": .08, "fan_in": .08,
         "fan_out": .07, "churn": .13, "bug_fixes": .15, "n_authors": .05, "todo_count": .07}
    score = sum(df[c].rank(pct=True) * wt for c, wt in w.items())
    return (100 * score / sum(w.values())).round(1)
