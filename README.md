# TechDebt-AI

Detects technical debt and predicts which files are likely to become hard to maintain.

## Setup
    pip install -r requirements.txt

## Run (needs a git repo with real history; more history = better model)
    python cli.py scan    /path/to/repo            # heuristic debt score only (no training)
    python cli.py train   /path/to/repo --horizon 180 --snapshots 4
    python cli.py predict /path/to/repo --top 25   # writes report.csv + report.html

## How it works
1. git_history.py   - mines commits: churn, authors, bug-fix commits (message heuristic)
2. static_metrics.py - lizard: cyclomatic complexity, function length, params; import graph -> fan-in/fan-out
3. features.py      - one row per file, built *as of a past cutoff date* (no leakage)
4. model.py         - label = "file got a bug-fix commit in the next N days"; RandomForest; train on older snapshots, test on the newest
5. suggest.py       - rule-based, explainable refactoring advice from each file's outlier metrics

## Ideas to extend
- Replace the commit-message bug heuristic with GitHub/Jira issue links or SZZ
- Add SonarQube/pylint warnings, test coverage, code-clone and code-embedding (CodeBERT) features
- Use SHAP for per-file explanations; swap in XGBoost/LightGBM
- Streamlit dashboard; GitHub Action that comments risk on pull requests
