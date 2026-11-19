"""Project 3, Session 3: the full-year check for Chronos-Bolt small and base.

Same setup as Project 2's full-year test (`full_year` in src/evaluate_baselines.py),
fixed before this run (19 Nov 2026):
  - "Full year" = the sum of the next 4 quarters from one forecast origin
    (h = 1, 2, 3, 4, all forecast at that origin). % error on the sum.
  - A window counts only if all 4 of its actuals were first published after the
    release date, 25 Nov 2024 (both models' repos were created that day).
  - Chronos small and base: median, the same quarters ETS used at each origin
    (65-day rule, `training_series`), forecasting up to 4 steps ahead. ETS
    (results/session4_forecasts.csv) and B4 (results/session7_b4_forecasts.csv)
    are the saved Project 2 forecasts for the same origins and targets.
  - Windows whose 4 quarters are exactly one fiscal year (origin = fiscal Q4)
    are flagged and also shown separately (descriptive).

Tests (as in Project 2, each family Holm-corrected):
  1. Across companies: sign tests, small and base vs ETS and vs B4, base vs small (5 tests).
  2. Per company: Diebold-Mariano with the Harvey correction on the full-year
     absolute % errors, lag 4 (neighbouring windows share 3 quarters). With 3-4
     windows the Harvey factor is zero, so those companies cannot be tested.

Run from the project folder:
    .venv/bin/python src/chronos_fullyear.py
Saves results/chronos_fullyear_forecasts.csv (quarters), chronos_fullyear_windows.csv,
chronos_fullyear_by_company.csv, chronos_fullyear_by_type.csv, chronos_fullyear_tests.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from baselines import HORIZONS, cutoff_date, known_at, load
from chronos_bolt import MODEL as SMALL, RELEASE_DATE, REVISION as SMALL_REV
from chronos_base import MODEL as BASE, REVISION as BASE_REV
from companies import COMPANIES, TYPES
from ets import training_series
from evaluate_ets import dm_test, holm
from evaluate_lgbm import sign_test

RESULTS = Path("results")
METHODS = ["small", "base", "ets", "b4"]
LAG = 4


def eligible_origins(cdf):
    """Origins whose next 4 quarters all exist and were all published after the release."""
    filed = cdf.set_index("quarter")["first_filed_date"]
    return [o for o in filed.index
            if all(o + h in filed.index and filed[o + h] > RELEASE_DATE for h in HORIZONS)]


def chronos_paths(pipelines, df):
    rows = []
    for company, cdf in df.groupby("company", sort=False):
        cdf = cdf.reset_index(drop=True)
        actual = cdf.set_index("quarter")["value"]
        for origin in eligible_origins(cdf):
            context = training_series(known_at(cdf, origin))
            assert (cdf.set_index("quarter").loc[context.index, "first_filed_date"]
                    <= cutoff_date(origin, cdf)).all(), "leak"
            steps = [(origin + h - context.index[-1]).n for h in HORIZONS]
            x = torch.tensor(context.to_numpy() / 1e9, dtype=torch.float32)
            med = {}
            for name, p in pipelines.items():
                q, _ = p.predict_quantiles(x, prediction_length=max(steps), quantile_levels=[0.5])
                med[name] = q[0, :, 0].numpy() * 1e9
            for h, s in zip(HORIZONS, steps):
                rows.append({"company": company, "type": COMPANIES[company][0], "origin": str(origin),
                             "h": h, "target": str(origin + h), "actual": actual[origin + h],
                             "steps": s, "n_context": len(context),
                             "small": med["small"][s - 1], "base": med["base"][s - 1]})
    return pd.DataFrame(rows)


def add_ets_b4(f):
    s4 = pd.read_csv(RESULTS / "session4_forecasts.csv")
    ets = s4[s4.baseline == "ETS"][["company", "origin", "h", "forecast"]].rename(columns={"forecast": "ets"})
    b4 = pd.read_csv(RESULTS / "session7_b4_forecasts.csv")[["company", "origin", "h", "b4"]]
    f = f.merge(ets, on=["company", "origin", "h"], how="left").merge(b4, on=["company", "origin", "h"], how="left")
    assert f[["ets", "b4"]].notna().all().all(), "missing ETS or B4 forecast"
    return f


def windows(f):
    w = (f.groupby(["company", "type", "origin"], sort=False)
          .agg(actual=("actual", "sum"), **{m: (m, "sum") for m in METHODS},
               first=("target", "first"), last=("target", "last"))
          .reset_index())
    w["window"] = w["first"] + "–" + w["last"]
    w["fiscal year"] = [f"FY{q[:4]}" if o.endswith("Q4") else "" for o, q in zip(w.origin, w["first"])]
    for m in METHODS:
        w[f"ape_{m}"] = 100 * (w[m] - w.actual).abs() / w.actual
        w[f"err_{m}"] = 100 * (w[m] - w.actual) / w.actual
    return w


def by_company(w):
    g = w.groupby("company", sort=False)
    t = pd.DataFrame({"type": g.type.first(), "windows": g.size(),
                      "first window": g.window.first(), "last window": g.window.last(),
                      "fiscal years": g["fiscal year"].apply(lambda s: ", ".join(x for x in s if x) or "none")})
    for m in METHODS:
        t[f"MAPE {m}"] = g[f"ape_{m}"].mean()
    for m in ["small", "base"]:
        for o in ["ets", "b4"]:
            t[f"{m} beats {o}"] = t[f"MAPE {m}"] < t[f"MAPE {o}"]
    t["base beats small"] = t["MAPE base"] < t["MAPE small"]
    t["small error % (+ high)"] = g.err_small.mean()
    t["base error % (+ high)"] = g.err_base.mean()
    return t


def by_type(t):
    rows = []
    for label, tt in [(ty, t[t.type == ty]) for ty in TYPES] + [("All 20", t)]:
        r = {"type": label, "companies": len(tt), "windows": int(tt.windows.sum())}
        for m in METHODS:
            r[f"MAPE {m}"] = tt[f"MAPE {m}"].mean()
        for c in ["small beats ets", "small beats b4", "base beats ets", "base beats b4", "base beats small"]:
            r[c] = f"{int(tt[c].sum())}/{len(tt)}"
        r["small too low"] = f"{int((tt['small error % (+ high)'] < 0).sum())}/{len(tt)}"
        r["base too low"] = f"{int((tt['base error % (+ high)'] < 0).sum())}/{len(tt)}"
        rows.append(r)
    return pd.DataFrame(rows).set_index("type")


def tests(t, w):
    pairs = [("small", "ets"), ("small", "b4"), ("base", "ets"), ("base", "b4"), ("base", "small")]
    rows = []
    for m, o in pairs:
        k = int((t[f"MAPE {m}"] < t[f"MAPE {o}"]).sum())
        rows.append({"family": "1 across companies (sign test)", "test": f"{m} vs {o}", "company": "all 20",
                     "n": len(t), "wins": k, "stat": np.nan, "p": sign_test(k, len(t))})
    for m, o in pairs:
        for company, d in w.groupby("company", sort=False):
            d = d.sort_values("origin")
            T = len(d)
            factor = (T + 1 - 2 * LAG + LAG * (LAG - 1) / T) / T
            stat, p = dm_test(d[f"ape_{m}"], d[f"ape_{o}"], LAG) if factor > 0 else (np.nan, np.nan)
            rows.append({"family": "2 per company (DM, Harvey, lag 4)", "test": f"{m} vs {o}", "company": company,
                         "n": T, "wins": int(d[f"ape_{m}"].mean() < d[f"ape_{o}"].mean()), "stat": stat, "p": p})
    r = pd.DataFrame(rows)
    for fam, idx in r.groupby("family").groups.items():
        ok = r.loc[idx, "p"].notna()
        r.loc[ok[ok].index, "Holm p"] = holm(r.loc[ok[ok].index, "p"])
    r["survives Holm (5 %)"] = r["Holm p"] < 0.05
    return r


def main():
    torch.manual_seed(0)
    pipelines = {name: BaseChronosPipeline.from_pretrained(model, revision=rev, device_map="cpu",
                                                           torch_dtype=torch.float32)
                 for name, model, rev in [("small", SMALL, SMALL_REV), ("base", BASE, BASE_REV)]}
    df = load(companies=list(COMPANIES))
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 60)

    f = add_ets_b4(chronos_paths(pipelines, df))
    f.to_csv(RESULTS / "chronos_fullyear_forecasts.csv", index=False)

    # Check: the h = 1 forecasts equal the one-quarter-ahead runs of Sessions 2 and 3.
    one = f[f.h == 1].merge(pd.read_csv(RESULTS / "chronos_all_forecasts.csv")[["company", "target", "chronos"]],
                            on=["company", "target"]).merge(
          pd.read_csv(RESULTS / "chronos_base_forecasts.csv")[["company", "target", "base"]],
          on=["company", "target"], suffixes=("", "_s3"))
    print(f"Checks: {len(f)} quarter forecasts in {f.groupby(['company', 'origin']).ngroups} windows; "
          f"h = 1 vs Session 2 small: max diff {(100 * (one.small - one.chronos).abs() / one.chronos).max():.1e} %, "
          f"vs Session 3 base: {(100 * (one.base - one.base_s3).abs() / one.base_s3).max():.1e} % "
          f"({len(one)} matched); origins with steps > h: {(f.steps > f.h).sum()} rows")

    w = windows(f)
    w.to_csv(RESULTS / "chronos_fullyear_windows.csv", index=False)
    t = by_company(w)
    t.to_csv(RESULTS / "chronos_fullyear_by_company.csv")
    s = by_type(t)
    s.to_csv(RESULTS / "chronos_fullyear_by_type.csv")
    r = tests(t, w)
    r.to_csv(RESULTS / "chronos_fullyear_tests.csv", index=False)

    print("\nWINDOWS THAT COUNT (all 4 actuals published after", RELEASE_DATE.date(), ")")
    print(t[["type", "windows", "first window", "last window", "fiscal years"]].to_string())
    print("\nONE ROW PER COMPANY (full-year MAPE, %)")
    print(t.drop(columns=["first window", "last window", "fiscal years"]).round(2).to_string())
    print("\nBY TYPE AND OVERALL (MAPE = average of the companies' MAPEs)")
    print(s.round(2).to_string())
    fy = w[w["fiscal year"] != ""]
    print(f"\nFISCAL-YEAR WINDOWS ONLY (descriptive): {len(fy)} windows, {fy.company.nunique()} companies; "
          "MAPE averaged over windows: " + ", ".join(f"{m} {fy[f'ape_{m}'].mean():.2f}" for m in METHODS))
    print("\nTESTS (Holm within each family)")
    print(r[r.family.str.startswith("1")].round(4).to_string(index=False))
    fam2 = r[r.family.str.startswith("2")]
    print(f"\nFamily 2: {len(fam2)} DM tests, {fam2.p.notna().sum()} testable; raw p < 0.05: "
          f"{(fam2.p < 0.05).sum()}; survive Holm: {fam2['survives Holm (5 %)'].sum()}")
    print(fam2.sort_values("p").head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
