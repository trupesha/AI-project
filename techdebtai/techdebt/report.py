"""CSV + HTML report."""
import html
import pandas as pd

SHOW = ["risk", "debt_score", "max_ccn", "nloc", "fan_in", "fan_out", "churn", "bug_fixes", "n_authors", "suggestions"]


def write_reports(df: pd.DataFrame, top: int, csv_path="report.csv", html_path="report.html"):
    df.to_csv(csv_path)
    t = df.head(top)[SHOW].copy()
    t["suggestions"] = t["suggestions"].map(html.escape)
    style = ("<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;font-size:13px}"
             "td,th{border:1px solid #ddd;padding:6px 8px;vertical-align:top}th{background:#f3f3f3}"
             "tr:hover{background:#fafafa}</style>")
    body = t.to_html(escape=False, float_format="%.2f")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'>{style}</head><body>"
                f"<h2>Technical-debt risk report (top {top})</h2>"
                f"<p>risk = predicted probability of bug-fix activity in the next window; "
                f"debt_score = 0-100 current-debt heuristic.</p>{body}</body></html>")
    print(f"Wrote {csv_path} and {html_path}")
