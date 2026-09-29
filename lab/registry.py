"""Every test you run, remembered — the laboratory's defence against overfitting (§9.7).

"Try fifty variants, keep the best" always finds something. The defence is to count the
fifty. Each evaluation appends one line per result to ``logs/lab_registro.jsonl``
(gitignored, like the rest of ``logs/``), and :func:`summary` applies Benjamini-Hochberg
over **every distinct test ever recorded** — the same rule at the same horizon with the same
parameters counts once, its latest run — not over today's handful.

A result that survives only when you forget the other tries has not survived.

Tests on the **holdout** (``lab.holdout_from``) are kept apart: the holdout is the one look
at data the rule never saw while it was being shaped. Looking at it repeatedly turns it into
more in-sample data; :func:`summary` says how many times it was used.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from validation.metrics import benjamini_hochberg

PATH = Path("logs/lab_registro.jsonl")


def _key(row: dict[str, Any]) -> str:
    spec = json.dumps({k: row.get(k) for k in ("rule", "kind", "horizon", "params")},
                      sort_keys=True, default=str)
    return hashlib.sha1(spec.encode()).hexdigest()[:12]


def record(results: Sequence[Any], *, holdout: bool = False, path: Path | None = None) -> None:
    """Append each result (``lab.evaluate.Result``) with a timestamp."""
    target = Path(path or PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with open(target, "a", encoding="utf-8") as handle:
        for r in results:
            row = r.row() if hasattr(r, "row") else dict(r)
            row.update({"when": stamp, "holdout": bool(holdout)})
            row["key"] = _key(row)
            handle.write(json.dumps(row, default=str) + "\n")


def load(path: Path | None = None) -> pd.DataFrame:
    target = Path(path or PATH)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_json(target, lines=True, dtype=False)


def summary(alpha: float = 0.10, path: Path | None = None) -> pd.DataFrame:
    """One row per distinct test (latest run), with BH q-values over **all** of them.

    Columns: rule, kind, horizon, n, mean, p, q, significant, runs (how many times that
    exact test was run), holdout. In-sample tests and holdout tests get separate BH
    families: they answer different questions.
    """
    rows = load(path)
    if rows.empty:
        return rows
    rows["runs"] = rows.groupby(["key", "holdout"])["key"].transform("size")
    latest = rows.sort_values("when").drop_duplicates(["key", "holdout"], keep="last")
    out = []
    for flag, part in latest.groupby("holdout"):
        part = part.copy()
        part["q"] = float("nan")
        part["significant"] = False
        tested = part["p"].notna()
        qs = benjamini_hochberg(part.loc[tested, "p"].astype(float).tolist(), alpha)
        part.loc[tested, "q"] = [q for q, _ in qs]
        part.loc[tested, "significant"] = [s for _, s in qs]
        out.append(part)
    frame = pd.concat(out, ignore_index=True)
    columns = ["rule", "kind", "horizon", "n", "mean", "hit", "p", "q", "significant",
               "half_1", "half_2", "runs", "holdout", "when", "params"]
    return frame.reindex(columns=columns).sort_values(["holdout", "q", "p"],
                                                      na_position="last")
