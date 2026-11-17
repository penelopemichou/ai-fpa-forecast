"""Session 7, Step 3: is smoothing the reason ETS wins?

B4 (src/smoothing.py) is computed on the same rolling grid as every model since
Session 4 and saved to results/session7_b4_forecasts.csv.

Main tests (fixed before evaluating), 20 companies, two-sided sign tests:
  1. h = 1:     in how many companies does B4 have a lower MAPE than ETS?
  2. full year: in how many companies does B4 beat ETS?
  Companies share shocks (2020 above all), so they aren't fully independent and
  the sign test is somewhat too optimistic.
Descriptive: B4 vs B3 and vs the old best simple rule (h = 1 and full year), by
type, and how much the 2020 steps inside B4's 3-year average hurt the 2021-2023
forecasts (a diagnostic only: B4 without the 2020 steps is NOT a competing rule).
Alphabet without the Motorola targets.

Run from the project folder:
    .venv/bin/python src/evaluate_smoothing.py run        # B4 forecasts + leak check
    .venv/bin/python src/evaluate_smoothing.py evaluate
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import HORIZONS, cutoff_date, first_origin, known_at, load
from companies import COMPANIES, TYPES, company_type
from evaluate_baselines import MOTOROLA, full_year
from evaluate_lgbm import sign_test
from evaluate_latest import all_methods as session6_methods
from evaluate_types import best_rules, load_forecasts
from smoothing import b4_latest_times_average_step, b4_steps

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "session7_b4_forecasts.csv"
ALL = list(COMPANIES)


# --- B4 on the rolling grid ---------------------------------------------------------

def rolling_b4(df=None):
    if df is None:
        df = load(companies=ALL)
    rows = []
    for company, cdf in df.groupby("company", sort=False):
        s = cdf.set_index("quarter")
        actual, ends, filed = s["value"], s["period_end"], s["first_filed_date"]
        last_quarter = actual.index.max()
        origin = first_origin(cdf)
        while origin + 1 <= last_quarter:
            known = known_at(cdf, origin)
            cutoff = cutoff_date(origin, cdf)
            for h in HORIZONS:
                target = origin + h
                if target not in actual.index:
                    continue
                steps = b4_steps(known, target)
                used = [known.index[-1]] + [q for a, b, _ in steps for q in (a, b)]
                # B4 with the 2020 steps left out (diagnostic only, see docstring)
                kept = [r for a, b, r in steps
                        if ends[a].year != 2020 and ends[b].year != 2020]
                no2020 = (float(known.iloc[-1]) * float(np.exp(np.mean(np.log(kept))))) if kept else np.nan
                rows.append({"company": company, "origin": origin, "h": h, "target": target,
                             "actual": actual[target], "b4": b4_latest_times_average_step(known, target),
                             "years_used": len(steps),
                             "has_2020_step": len(kept) < len(steps), "b4_without_2020": no2020,
                             "cutoff": cutoff, "latest_used_filed": max(filed[q] for q in used)})
            origin += 1
    return pd.DataFrame(rows)


def leak_check(f):
    return pd.Series({
        "forecasts": len(f),
        "uses a quarter public after the cutoff": int((f["latest_used_filed"] > f["cutoff"]).sum()),
        "target not after the origin": int((f["target"] <= f["origin"]).sum()),
        "missing B4": int(f["b4"].isna().sum()),
    })


def run():
    f = rolling_b4()
    OUT.parent.mkdir(exist_ok=True)
    f.to_csv(OUT, index=False)
    print(f"Saved {len(f):,} B4 forecasts to {OUT.relative_to(PROJECT)}")
    print("Years of steps used:", f["years_used"].value_counts().sort_index().to_dict())
    print("Forecasts whose 3-year average includes a 2020 step:", int(f["has_2020_step"].sum()), "\n")
    print("Leak check (all should be 0 except the count)")
    print(leak_check(f).to_string())


# --- Evaluation -----------------------------------------------------------------------

def load_b4():
    g = pd.read_csv(OUT)
    for col in ["origin", "target"]:
        g[col] = pd.PeriodIndex(g[col], freq="Q")
    return g[~((g["company"] == "Alphabet") & g["target"].between(*MOTOROLA))]


def all_methods():
    """Session 6's table (B0-B3, ETS, LightGBM, ...) plus B4."""
    base = session6_methods()
    g = load_b4()
    keys = ["company", "origin", "h"]
    extra = base[base["baseline"] == "ETS"][keys + ["mase_scale", "type", "target_end", "year_2020", "motorola"]]
    e = g[keys + ["target", "actual", "b4"]].merge(extra, on=keys)
    e["baseline"], e["forecast"] = "B4", e["b4"]
    e["ape"] = 100 * (e["actual"] - e["forecast"]).abs() / e["actual"]
    e["ase"] = (e["actual"] - e["forecast"]).abs() / e["mase_scale"]
    return pd.concat([base, e[base.columns]], ignore_index=True)


def per_company(f, rules):
    h1, fy = f[f["h"] == 1], full_year(f)
    m1 = h1.pivot_table(index="company", columns="baseline", values="ape").reindex(ALL)
    mf = fy.pivot_table(index="company", columns="baseline", values="ape").reindex(ALL)
    t = pd.DataFrame({"type": [company_type(c) for c in ALL], "best rule": rules["best"]}, index=ALL)
    for label, m in [("h1", m1), ("FY", mf)]:
        t[f"{label} B4"], t[f"{label} ETS"], t[f"{label} B3"] = m["B4"], m["ETS"], m["B3"]
        t[f"{label} best rule"] = [m.loc[c, rules.loc[c, "best"]] for c in ALL]
        for other in ["ETS", "B3", "best rule"]:
            t[f"{label}: B4 beats {other}"] = t[f"{label} B4"] < t[f"{label} {other}"]
    return t


def win_table(t):
    rows = []
    for label in ["h1", "FY"]:
        for other in ["ETS", "B3", "best rule"]:
            col = f"{label}: B4 beats {other}"
            w = int(t[col].sum())
            row = {"comparison": f"{'one quarter ahead' if label == 'h1' else 'full year'}: B4 beats {other}",
                   "wins": f"{w}/20", "sign test p (two-sided)": sign_test(w, 20)}
            row.update({ty: f"{int(t[t['type'] == ty][col].sum())}/5" for ty in TYPES})
            rows.append(row)
    return pd.DataFrame(rows).set_index("comparison")


def main_tests(t):
    return win_table(t).loc[["one quarter ahead: B4 beats ETS", "full year: B4 beats ETS"],
                            ["wins", "sign test p (two-sided)"]]


def by_type(t):
    cols = ["h1 ETS", "h1 B4", "h1 B3", "h1 best rule", "FY ETS", "FY B4", "FY B3", "FY best rule"]
    return t.groupby("type")[cols].mean().reindex(TYPES)


def all_horizons(f):
    m = f.pivot_table(index="h", columns="baseline", values="ape", aggfunc="mean")
    return m[["ETS", "B4", "B3", "B2", "B1", "B0"]]


def effect_of_2020():
    """For h = 1 forecasts whose 3-year average includes a 2020 step: MAPE of B4 as
    defined vs the same average with the 2020 steps left out (diagnostic only), by type."""
    g = load_b4()
    g = g[(g["h"] == 1) & g["has_2020_step"] & g["b4_without_2020"].notna()].copy()
    g["type"] = g["company"].map(company_type)
    g["B4"] = 100 * (g["actual"] - g["b4"]).abs() / g["actual"]
    g["B4 without 2020 steps"] = 100 * (g["actual"] - g["b4_without_2020"]).abs() / g["actual"]
    out = g.groupby("type")[["B4", "B4 without 2020 steps"]].mean().reindex(TYPES)
    out["forecasts"] = g.groupby("type").size().reindex(TYPES)
    out["cost of the 2020 steps (points)"] = out["B4"] - out["B4 without 2020 steps"]
    out.loc["all"] = [g["B4"].mean(), g["B4 without 2020 steps"].mean(), len(g),
                      g["B4"].mean() - g["B4 without 2020 steps"].mean()]
    return out


def without_2020_diagnostic(f):
    """POST-HOC DIAGNOSTIC, found after seeing the results: at h = 1, how would B4
    compare with ETS if the 2020 steps were left out of its average? Not a test and
    not a competing rule; a hypothesis for later only."""
    g = load_b4()
    g = g[g["h"] == 1].copy()
    g["b4x"] = np.where(g["has_2020_step"] & g["b4_without_2020"].notna(), g["b4_without_2020"], g["b4"])
    g["ape B4"] = 100 * (g["actual"] - g["b4"]).abs() / g["actual"]
    g["ape B4 without 2020 steps"] = 100 * (g["actual"] - g["b4x"]).abs() / g["actual"]
    ets = f[(f["h"] == 1) & (f["baseline"] == "ETS")].groupby("company")["ape"].mean()
    per = g.groupby("company")[["ape B4", "ape B4 without 2020 steps"]].mean().reindex(ALL)
    per["ETS"] = ets.reindex(ALL)
    return pd.Series({
        "mean h1 MAPE, B4": g["ape B4"].mean(),
        "mean h1 MAPE, B4 without 2020 steps": g["ape B4 without 2020 steps"].mean(),
        "mean h1 MAPE, ETS": f[(f["h"] == 1) & (f["baseline"] == "ETS")]["ape"].mean(),
        "companies where B4 beats ETS": int((per["ape B4"] < per["ETS"]).sum()),
        "companies where B4 without 2020 steps beats ETS": int((per["ape B4 without 2020 steps"] < per["ETS"]).sum()),
    })


def year_2020(f):
    g = f[(f["h"] == 1) & f["baseline"].isin(["ETS", "B3", "B4"])].assign(
        when=lambda d: np.where(pd.to_datetime(d["target_end"]).dt.year == 2020, "2020", "other years"))
    return g.pivot_table(index="type", columns=["baseline", "when"], values="ape", aggfunc="mean").reindex(TYPES)


def evaluate():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    f = all_methods()
    t = per_company(f, best_rules(load_forecasts()))
    print("MAIN TESTS (sign test two-sided; companies share shocks, so p is somewhat optimistic)")
    print(main_tests(t).to_string(), "\n")
    print("All comparisons, with wins by type")
    print(win_table(t).to_string(), "\n")
    print("Per company (MAPE %)")
    print(t[[c for c in t.columns if ":" not in c]].to_string(), "\n")
    print("By type (mean MAPE %)")
    print(by_type(t).to_string(), "\n")
    print("Mean MAPE over all forecasts, %")
    print(all_horizons(f).to_string(), "\n")
    print("2020 steps inside the 3-year average (h = 1, 2021-2023 forecasts): B4 vs B4 without them")
    print(effect_of_2020().to_string(), "\n")
    print("h = 1 MAPE, targets ending in calendar 2020 vs other years")
    print(year_2020(f).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    elif sys.argv[1:] == ["evaluate"]:
        evaluate()
    else:
        print(__doc__)
