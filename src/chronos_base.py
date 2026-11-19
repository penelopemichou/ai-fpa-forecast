"""Project 3, Session 3: is a bigger model better? Chronos-Bolt base on all 20 companies.

Only one change from Session 2 (src/chronos_all.py), fixed before this run (19 Nov 2026):
  - amazon/chronos-bolt-base (about 205M parameters) instead of chronos-bolt-small
    (48M), pinned revision below. Its Hugging Face repo was created on 25 Nov 2024,
    the same day as small's, so the same release-date rule and the same
    158 quarters apply.
  - Everything else is the same: median, one quarter ahead, 65-day rule, the
    same quarters ETS used at each origin, ETS refit and B4 as in Session 2.
  - New: small's forecasts (results/chronos_all_forecasts.csv, untouched) are
    compared with base's on exactly the same quarters.

Tests (as in Session 2, each family Holm-corrected):
  1. Across companies: sign tests, base vs ETS, vs B4 and vs small (3 tests).
  2. Per company: Diebold-Mariano with the Harvey correction, base vs ETS, vs B4
     and vs small (60 tests).

Seasonal lean: where base sits between B1, same quarter last year (0), and
B0, last quarter (1): (forecast - B1) / (B0 - B1), median per type.

Run from the project folder:
    .venv/bin/python src/chronos_base.py
Saves results/chronos_base_forecasts.csv, chronos_base_by_company.csv,
chronos_base_by_type.csv, chronos_base_lean.csv and chronos_base_tests.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from baselines import load
from chronos_all import by_company, by_type, forecasts
from chronos_bolt import H
from companies import COMPANIES, TYPES
from evaluate_ets import dm_test, holm
from evaluate_lgbm import sign_test

MODEL = "amazon/chronos-bolt-base"
REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"   # latest on 19 Nov 2026
RESULTS = Path("results")


def add_small(f):
    small = pd.read_csv(RESULTS / "chronos_all_forecasts.csv")[["company", "target", "chronos", "ape_chronos"]]
    small = small.rename(columns={"chronos": "small", "ape_chronos": "ape_small"})
    keys = lambda d: set(zip(d.company, d.target))
    assert keys(f) == keys(small), "base and small are not scored on the same quarters"
    return f.merge(small, on=["company", "target"])


def lean(f):
    """Median position between B1 (0) and B0 (1), per type, for base, small and ETS."""
    s4 = pd.read_csv(RESULTS / "session4_forecasts.csv")
    s4 = s4[(s4.h == H) & s4.baseline.isin(["B0", "B1"])]
    p = s4.pivot_table(index=["company", "target"], columns="baseline", values="forecast").reset_index()
    m = f.merge(p, on=["company", "target"])
    assert len(m) == len(f)
    for col, name in [("chronos", "base"), ("small", "small"), ("ets", "ETS")]:
        m[name] = (m[col] - m.B1) / (m.B0 - m.B1)
    out = m.groupby("type")[["base", "small", "ETS"]].median().reindex(TYPES)
    out.loc["All 20"] = m[["base", "small", "ETS"]].median()
    return out


def tests(t, f):
    rows = []
    for other in ["ets", "b4", "small"]:
        w = int((t["MAPE chronos"] < t[f"MAPE {other}"]).sum())
        rows.append({"family": "1 across companies (sign test)", "test": f"base vs {other}",
                     "company": "all 20", "n": len(t), "base wins": w, "stat": np.nan,
                     "p": sign_test(w, len(t))})
    for other in ["ets", "b4", "small"]:
        for company, d in f.groupby("company", sort=False):
            d = d.sort_values("target")
            stat, p = dm_test(d.ape_chronos, d[f"ape_{other}"], H)
            rows.append({"family": "2 per company (DM, Harvey)", "test": f"base vs {other}",
                         "company": company, "n": len(d),
                         "base wins": int(d.ape_chronos.mean() < d[f"ape_{other}"].mean()),
                         "stat": stat, "p": p})
    r = pd.DataFrame(rows)
    for fam, idx in r.groupby("family").groups.items():
        r.loc[idx, "Holm p"] = holm(r.loc[idx, "p"])
    r["survives Holm (5 %)"] = r["Holm p"] < 0.05
    return r


def as_base(d):
    """Column names say 'base' instead of 'chronos' in the saved and printed tables."""
    return d.rename(columns=lambda c: c.replace("chronos", "base"))


def main():
    torch.manual_seed(0)
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL, revision=REVISION, device_map="cpu", torch_dtype=torch.float32)
    print("parameters:", sum(v.numel() for v in pipeline.model.parameters()))
    df = load(companies=list(COMPANIES))
    pd.set_option("display.width", 240, "display.max_columns", 30)

    f = add_small(forecasts(pipeline, df))
    as_base(f).to_csv(RESULTS / "chronos_base_forecasts.csv", index=False)
    print(f"Checks: {len(f)} forecasts, the same quarters as small; Chronos context = ETS "
          f"training quarters; largest ETS refit vs Project 2 difference: {f.ets_refit_diff_pct.max():.2e} %")

    t = by_company(f)
    t.insert(t.columns.get_loc("MAPE chronos") + 1, "MAPE small",
             f.groupby("company", sort=False).ape_small.mean())
    as_base(t).to_csv(RESULTS / "chronos_base_by_company.csv")
    print("\n2. ONE ROW PER COMPANY (MAPE and ranges in %)")
    print(as_base(t).round(2).to_string())

    s = by_type(t, f)
    small_by_type = t.groupby("type")["MAPE small"].mean().reindex(TYPES)
    small_by_type["All 20"] = t["MAPE small"].mean()
    s.insert(s.columns.get_loc("MAPE chronos") + 1, "MAPE small", small_by_type)
    s.insert(s.columns.get_loc("MAPE b4") + 1, "base beats small",
             [f"{(g['MAPE chronos'] < g['MAPE small']).sum()}/{len(g)}" for g in
              [t[t.type == ty] for ty in TYPES] + [t]])
    as_base(s).to_csv(RESULTS / "chronos_base_by_type.csv")
    print("\nBY TYPE AND OVERALL (MAPE = average of the companies' MAPEs; ranges over quarters)")
    print(as_base(s).round(2).to_string())

    l = lean(f)
    l.to_csv(RESULTS / "chronos_base_lean.csv")
    print("\n3. SEASONAL LEAN (median; 0 = same quarter last year, 1 = last quarter)")
    print(l.round(2).to_string())

    r = tests(t, f)
    r.to_csv(RESULTS / "chronos_base_tests.csv", index=False)
    print("\n4. TESTS (Holm within each family)")
    print(r[r.family.str.startswith("1")].round(4).to_string(index=False))
    fam2 = r[r.family.str.startswith("2")]
    print(f"\nFamily 2: {len(fam2)} DM tests; raw p < 0.05: {(fam2.p < 0.05).sum()}; survive Holm: "
          f"{fam2['survives Holm (5 %)'].sum()}")
    print(fam2.sort_values("p").head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
