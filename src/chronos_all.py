"""Project 3, Session 2: Chronos-Bolt small on all 20 companies.

Same setup as Session 1 (src/chronos_bolt.py), fixed before this run (19 Nov 2026):
  - amazon/chronos-bolt-small, pinned revision, CPU, float32, median forecast.
  - One quarter ahead, 65-day rule, the same quarters ETS used at each origin.
  - A target quarter counts only if its actual was first published (first_filed_date)
    after the release date, 25 Nov 2024. The rule uses each company's own filing
    dates, so fiscal years ending in other months are handled the same way.
  - Compared with ETS and B4 on exactly the same quarters. ETS is refitted at
    the scored origins (to get its 80 % range) and checked against the Project 2
    ETS forecasts in results/session4_forecasts.csv; B4 comes from
    results/session7_b4_forecasts.csv.

Tests (the two kinds used in Project 2), each family Holm-corrected:
  1. Across companies (Sessions 4-7): in how many of the 20 companies does Chronos
     have the lower MAPE? Two-sided sign test, vs ETS and vs B4 (2 tests).
  2. Per company (Sessions 2-3): Diebold-Mariano with the Harvey correction on the
     absolute % errors, h = 1, vs ETS and vs B4 (40 tests).

Run from the project folder:
    .venv/bin/python src/chronos_all.py
Saves results/chronos_all_forecasts.csv, chronos_all_by_company.csv,
chronos_all_by_type.csv and chronos_all_tests.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from baselines import cutoff_date, known_at, load
from chronos_bolt import H, MODEL, RELEASE_DATE, REVISION, chronos_forecasts, scored_targets
from companies import COMPANIES, TYPES
from ets import ets_at_origin, training_series
from evaluate_ets import dm_test, holm
from evaluate_lgbm import sign_test

RESULTS = Path("results")
METHODS = ["chronos", "ets", "b4"]


def rule_check(df):
    """Per company: the last quarter left out and the first quarter scored, with filing dates."""
    rows = []
    for company, cdf in df.groupby("company", sort=False):
        c = cdf.set_index("quarter")
        targets = scored_targets(cdf.reset_index(drop=True))
        before = c[c["first_filed_date"] <= RELEASE_DATE]
        rows.append({"company": company,
                     "last left out": f"{before.index[-1]} (ended {before['period_end'].iloc[-1].date()}, "
                                      f"filed {before['first_filed_date'].iloc[-1].date()})",
                     "first scored": f"{targets[0]} (ended {c.loc[targets[0], 'period_end'].date()}, "
                                     f"filed {c.loc[targets[0], 'first_filed_date'].date()})",
                     "last scored": str(targets[-1]), "quarters": len(targets),
                     "all later quarters scored": all(c.loc[c.index > targets[0], "first_filed_date"] > RELEASE_DATE)})
    return pd.DataFrame(rows)


def forecasts(pipeline, df):
    out = []
    for company, cdf in df.groupby("company", sort=False):
        cdf = cdf.reset_index(drop=True)
        ch = chronos_forecasts(pipeline, cdf, company)
        actual = cdf.set_index("quarter")["value"]
        ets_rows, steps = [], []
        for target in scored_targets(cdf):
            known = known_at(cdf, target - H)
            steps.append((target - training_series(known).index[-1]).n)
            ets_rows += ets_at_origin(known, [target])
        ets = pd.DataFrame(ets_rows)
        ch["steps"] = steps
        ch["ets"], ch["ets_lo80"], ch["ets_hi80"] = ets["ets"].values, ets["lo80"].values, ets["hi80"].values
        ch["ets_n_train"] = ets["n_train"].values
        out.append(ch)
    f = pd.concat(out, ignore_index=True)
    assert (f.ets_n_train == f.n_context).all(), "Chronos and ETS saw different quarters"

    # ETS refit = Project 2's ETS forecasts?
    s4 = pd.read_csv(RESULTS / "session4_forecasts.csv")
    s4 = s4[(s4.baseline == "ETS") & (s4.h == H)][["company", "target", "forecast"]]
    f = f.merge(s4.rename(columns={"forecast": "ets_project2"}), on=["company", "target"], how="left")
    assert f.ets_project2.notna().all(), "a scored quarter has no Project 2 ETS forecast"
    f["ets_refit_diff_pct"] = 100 * (f.ets - f.ets_project2).abs() / f.ets_project2

    b4 = pd.read_csv(RESULTS / "session7_b4_forecasts.csv")
    b4 = b4[b4.h == H][["company", "target", "b4"]]
    f = f.merge(b4, on=["company", "target"], how="left")
    assert f.b4.notna().all(), "a scored quarter has no B4 forecast"

    f["type"] = f.company.map(lambda c: COMPANIES[c][0])
    for m in METHODS:
        f[f"ape_{m}"] = 100 * (f[m] - f.actual).abs() / f.actual
    f["err_chronos_pct"] = 100 * (f.chronos - f.actual) / f.actual      # + = too high
    f["in_chronos_10_90"] = (f.actual >= f.chronos_q10) & (f.actual <= f.chronos_q90)
    f["in_ets_80"] = (f.actual >= f.ets_lo80) & (f.actual <= f.ets_hi80)
    return f


def by_company(f):
    g = f.groupby("company", sort=False)
    t = pd.DataFrame({
        "type": g.type.first(), "quarters": g.size(),
        "MAPE chronos": g.ape_chronos.mean(), "MAPE ets": g.ape_ets.mean(), "MAPE b4": g.ape_b4.mean(),
        "chronos closer than ETS (quarters)": g.apply(lambda d: f"{(d.ape_chronos < d.ape_ets).sum()}/{len(d)}"),
        "chronos closer than B4 (quarters)": g.apply(lambda d: f"{(d.ape_chronos < d.ape_b4).sum()}/{len(d)}"),
        "inside chronos 10-90 %": 100 * g.in_chronos_10_90.mean(),
        "inside ETS 80 %": 100 * g.in_ets_80.mean(),
        "chronos mean error % (+ high)": g.err_chronos_pct.mean(),
    })
    t["chronos beats ETS"] = t["MAPE chronos"] < t["MAPE ets"]
    t["chronos beats B4"] = t["MAPE chronos"] < t["MAPE b4"]
    return t


def by_type(t, f):
    rows = []
    for label, sel_t, sel_f in [(ty, t.type == ty, f.type == ty) for ty in TYPES] + \
                               [("All 20", t.type.notna(), f.type.notna())]:
        tt, ff = t[sel_t], f[sel_f]
        rows.append({"type": label, "companies": len(tt), "quarters": len(ff),
                     "MAPE chronos": tt["MAPE chronos"].mean(), "MAPE ets": tt["MAPE ets"].mean(),
                     "MAPE b4": tt["MAPE b4"].mean(),
                     "chronos beats ETS": f"{tt['chronos beats ETS'].sum()}/{len(tt)}",
                     "chronos beats B4": f"{tt['chronos beats B4'].sum()}/{len(tt)}",
                     "inside chronos 10-90 %": 100 * ff.in_chronos_10_90.mean(),
                     "inside ETS 80 %": 100 * ff.in_ets_80.mean(),
                     "chronos too low (companies)": f"{(tt['chronos mean error % (+ high)'] < 0).sum()}/{len(tt)}",
                     "chronos mean error % (+ high)": ff.err_chronos_pct.mean()})
    return pd.DataFrame(rows).set_index("type")


def tests(t, f):
    rows = []
    for other in ["ets", "b4"]:
        w = int(t[f"chronos beats {other.upper() if other == 'ets' else 'B4'}"].sum())
        rows.append({"family": "1 across companies (sign test)", "test": f"chronos vs {other}",
                     "company": "all 20", "n": len(t), "chronos wins": w, "stat": np.nan,
                     "p": sign_test(w, len(t))})
    for other in ["ets", "b4"]:
        for company, d in f.groupby("company", sort=False):
            d = d.sort_values("target")
            stat, p = dm_test(d.ape_chronos, d[f"ape_{other}"], H)
            rows.append({"family": "2 per company (DM, Harvey)", "test": f"chronos vs {other}",
                         "company": company, "n": len(d),
                         "chronos wins": int(d.ape_chronos.mean() < d[f"ape_{other}"].mean()),
                         "stat": stat, "p": p})
    r = pd.DataFrame(rows)
    for fam, idx in r.groupby("family").groups.items():
        r.loc[idx, "Holm p"] = holm(r.loc[idx, "p"])
    r["survives Holm (5 %)"] = r["Holm p"] < 0.05
    return r


def main():
    torch.manual_seed(0)
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL, revision=REVISION, device_map="cpu", torch_dtype=torch.float32)
    df = load(companies=list(COMPANIES))
    pd.set_option("display.width", 220, "display.max_columns", 30)

    print("1. RULE CHECK: actual first published after", RELEASE_DATE.date())
    print(rule_check(df).to_string(index=False))

    f = forecasts(pipeline, df)
    f.to_csv(RESULTS / "chronos_all_forecasts.csv", index=False)
    print(f"\nChecks: {len(f)} forecasts; Chronos context = ETS training quarters at every origin; "
          f"origins where the origin quarter was not yet public (steps > 1): {(f.steps > 1).sum()}; "
          f"largest ETS refit vs Project 2 difference: {f.ets_refit_diff_pct.max():.2e} %")

    t = by_company(f)
    t.to_csv(RESULTS / "chronos_all_by_company.csv")
    print("\n2. ONE ROW PER COMPANY (MAPE and ranges in %)")
    print(t.round(2).to_string())

    s = by_type(t, f)
    s.to_csv(RESULTS / "chronos_all_by_type.csv")
    print("\n3. BY TYPE AND OVERALL (MAPE = average of the companies' MAPEs; ranges over quarters)")
    print(s.round(2).to_string())

    r = tests(t, f)
    r.to_csv(RESULTS / "chronos_all_tests.csv", index=False)
    print("\n4. TESTS (Holm within each family)")
    print(r[r.family.str.startswith("1")].round(4).to_string(index=False))
    fam2 = r[r.family.str.startswith("2")]
    print(f"\nFamily 2: {len(fam2)} DM tests; raw p < 0.05: {(fam2.p < 0.05).sum()}; survive Holm: "
          f"{fam2['survives Holm (5 %)'].sum()}")
    print(fam2.sort_values("p").head(6).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
