"""Usage:
  python cli.py train   /path/to/repo [--horizon 180] [--snapshots 4]
  python cli.py predict /path/to/repo [--top 25]
  python cli.py scan    /path/to/repo [--top 25]     # no ML, heuristic debt score only
"""
import argparse

from techdebt.features import build_snapshot, heuristic_debt_score
from techdebt.git_history import load_history
from techdebt.model import predict, train
from techdebt.report import write_reports
from techdebt.suggest import add_suggestions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "scan"])
    ap.add_argument("repo")
    ap.add_argument("--horizon", type=int, default=180, help="prediction window in days")
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    if a.cmd == "train":
        train(a.repo, a.model, a.horizon, a.snapshots)
        return

    if a.cmd == "predict":
        df = predict(a.repo, a.model)
    else:
        hist = load_history(a.repo)
        df = build_snapshot(a.repo, hist, int(hist.ts.max()) + 1, rev="HEAD")
        df["risk"] = float("nan")
    df["debt_score"] = heuristic_debt_score(df)
    df = add_suggestions(df).sort_values("risk" if a.cmd == "predict" else "debt_score", ascending=False)
    print(df.head(a.top)[["risk", "debt_score", "max_ccn", "churn", "bug_fixes", "suggestions"]].to_string())
    write_reports(df, a.top)


if __name__ == "__main__":
    main()
