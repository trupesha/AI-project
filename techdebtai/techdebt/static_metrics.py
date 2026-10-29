"""Static code metrics (complexity, size, dependencies) for any snapshot of the repo."""
import re
import subprocess
from collections import Counter
from pathlib import Path

import lizard
import pandas as pd

from .git_history import git

DEFAULT_EXTS = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rb", ".php", ".cs", ".kt"}
SKIP_PARTS = ("node_modules/", "vendor/", "/dist/", "/build/", "migrations/", ".min.")

IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+([\w\.]+)\s+import", re.M),            # python
    re.compile(r"^\s*import\s+([\w\.]+)", re.M),                   # python / java
    re.compile(r"^\s*#include\s+[<\"]([\w\./]+)[>\"]", re.M),      # c / c++
    re.compile(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)"),          # node
    re.compile(r"from\s+['\"]([^'\"]+)['\"]", re.M),               # es modules
]
DEBT_MARKERS = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")


def extract_imports(code: str) -> set:
    mods = set()
    for pat in IMPORT_PATTERNS:
        mods.update(pat.findall(code))
    return mods


def analyze_snapshot(repo: str, rev: str, exts=DEFAULT_EXTS, max_chars: int = 300_000) -> pd.DataFrame:
    """Compute metrics for every source file at git revision `rev` (no checkout needed)."""
    paths = [
        p for p in git(repo, "ls-tree", "-r", "--name-only", rev).splitlines()
        if Path(p).suffix in exts and not any(s in p for s in SKIP_PARTS)
    ]
    rows, imports = [], {}
    for p in paths:
        try:
            code = git(repo, "show", f"{rev}:{p}")
        except subprocess.CalledProcessError:
            continue
        if not code.strip() or len(code) > max_chars:
            continue
        info = lizard.analyze_file.analyze_source_code(p, code)
        funcs = info.function_list
        ccn = [f.cyclomatic_complexity for f in funcs] or [0]
        imports[p] = extract_imports(code)
        rows.append({
            "path": p,
            "nloc": info.nloc,
            "n_functions": len(funcs),
            "avg_ccn": sum(ccn) / len(ccn),
            "max_ccn": max(ccn),
            "avg_params": (sum(f.parameter_count for f in funcs) / len(funcs)) if funcs else 0,
            "max_func_nloc": max([f.nloc for f in funcs] or [0]),
            "todo_count": len(DEBT_MARKERS.findall(code)),
            "fan_out": len(imports[p]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Fan-in: how many other files import this file's module name (heuristic by file stem)
    stems = {}
    for p in df.path:
        stems.setdefault(Path(p).stem, []).append(p)
    fan_in = Counter()
    for p, mods in imports.items():
        for m in mods:
            tail = re.split(r"[./\\]", m)[-1]
            for target in stems.get(tail, []):
                if target != p:
                    fan_in[target] += 1
    df["fan_in"] = df.path.map(lambda x: fan_in.get(x, 0))
    return df.set_index("path")
