"""Session 3, Step 3: ARIMA and the combination (COMB) vs the current bars.

Main tests (fixed before seeing any results), 12 in total, Holm-corrected together:
  h = 1:     ARIMA vs ETS, COMB vs ETS                          (x 3 companies)
  full year: ARIMA vs best simple rule, COMB vs best simple rule (x 3 companies)
Alphabet in the main tests is without the Motorola targets; Alphabet with them
is shown for reference, outside the Holm family. Everything else is descriptive.

Run from the project folder:
    .venv/bin/python src/evaluate_arima.py
Saves results/arima_forecasts.csv, results/comb_forecasts.csv, results/arima_main_tests.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from arima import in_disguise, rolling_arima
from baselines import load
from combine import BEST_SIMPLE, rolling_combination
from ets import rolling_ets
from evaluate_baselines import (GROUPS, MOTOROLA, errors as baseline_errors, full_year,
                                metric_table, table_2020, with_groups)
from evaluate_ets import coverage, dm_test, holm, leak_check

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
BAR_SIMPLE = {**BEST_SIMPLE, "Alphabet excl. Motorola": BEST_SIMPLE["Alphabet"]}


# --- 1. All methods in one long table ----------------------------------------------

def to_long(df, column, name, scale):
    """Turn one method's forecasts into the same long format as the baselines."""
    e = df.merge(scale, on=["company", "origin"])
    e["baseline"], e["forecast"] = name, e[column]
    e["ape"] = 100 * (e["actual"] - e["forecast"]).abs() / e["actual"]
    e["ase"] = (e["actual"] - e["forecast"]).abs() / e["mase_scale"]
    e["motorola"] = (e["company"] == "Alphabet") & e["target"].between(*MOTOROLA)
    e["year_2020"] = e["target"].dt.year == 2020
    return e


def all_errors(value_col):
    """Baselines + ETS + ARIMA + COMB (column 'baseline' = method name)."""
    df = load(value_col)
    base = baseline_errors(value_col)
    scale = base.drop_duplicates(["company", "origin"])[["company", "origin", "mase_scale"]]
    ets = rolling_ets(df)
    arima = rolling_arima(df)
    comb = rolling_combination(df, ets=ets)
    parts = [base] + [to_long(d, col, name, scale)[base.columns]
                      for d, col, name in [(ets, "ets", "ETS"), (arima, "arima", "ARIMA"),
                                           (comb, "comb", "COMB")]]
    return pd.concat(parts, ignore_index=True), ets, arima, comb


# --- 2. Main tests -------------------------------------------------------------------

def compare(data, group, new, ref, lag, test):
    g = data[data["company"] == group]
    pair = g.pivot_table(index="origin", columns="baseline", values="ape").sort_index()
    a, b = pair[new], pair[ref]
    dm, p = dm_test(a, b, lag)
    return {"test": test, "company": group, "new": new, "vs": ref, "n": len(pair),
            "MAPE new": a.mean(), "MAPE vs": b.mean(),
            "improvement %": 100 * (b.mean() - a.mean()) / b.mean(),
            "new closer, % of forecasts": 100 * (a < b).mean(),
            "DM (HLN)": dm, "p-value": p}


def main_tests(long):
    h1 = with_groups(long[long["h"] == 1])
    fy = with_groups(full_year(long))
    rows = []
    for group in GROUPS:
        for new in ["ARIMA", "COMB"]:
            rows.append(compare(h1, group, new, "ETS", 1, "h=1"))
            rows.append(compare(fy, group, new, BAR_SIMPLE[group], 4, "full year"))
    tests = pd.DataFrame(rows).set_index(["test", "company", "new"]).sort_index(
        level=["test", "company"], key=lambda s: s.map({**{g: i for i, g in enumerate(GROUPS)},
                                                        "h=1": 0, "full year": 1}).fillna(0))
    family = tests.index.get_level_values("company") != "Alphabet"
    tests.loc[family, "Holm p"] = holm(tests.loc[family, "p-value"])
    return tests


# --- 3. Descriptive tables -------------------------------------------------------------

def order_choices(arima, top=4):
    """Most-chosen ARIMA orders per company, switches, and how often it was a simple rule."""
    per_origin = arima[arima["h"] == 1].sort_values(["company", "origin"])
    rows = []
    for company, g in per_origin.groupby("company", sort=False):
        counts = g["order"].value_counts()
        disguised = g["order"].map(in_disguise).ne("")
        rows.append({"company": company, "origins": len(g), "different orders": len(counts),
                     "switches": int((g["order"] != g["order"].shift()).sum() - 1),
                     "simple rule in disguise": int(disguised.sum()),
                     "most chosen": "; ".join(f"{o} x{n}" for o, n in counts.head(top).items())})
    return pd.DataFrame(rows).set_index("company")


def disguise_check(arima, df):
    """Where ARIMA chose a pure simple rule without drift, its forecast must equal that rule."""
    from baselines import rolling_forecasts
    base = rolling_forecasts(df)
    m = arima.merge(base, on=["company", "origin", "h"])
    out = {}
    for order, rule in [("(0,1,0)(0,0,0)4", "B0"), ("(0,0,0)(0,1,0)4", "B1")]:
        sel = m[m["order"] == order]
        out[f"{order} = {rule}: forecasts"] = len(sel)
        out[f"{order} = {rule}: largest difference, %"] = (
            float((100 * (sel["arima"] / sel[rule] - 1)).abs().max()) if len(sel) else np.nan)
    return pd.Series(out)


METHODS = ["B0", "B1", "B2", "ETS", "ARIMA", "COMB"]


def methods_table(long, metric):
    """metric_table with the methods in a fixed order."""
    return metric_table(long, metric).reindex(pd.MultiIndex.from_product([GROUPS, METHODS]))


def table_2020_methods(long):
    """2020 vs other years for ETS, ARIMA, COMB and each company's best simple rule."""
    t = table_2020(long)
    keep = [(c, m) for c in t.index.get_level_values(0).unique()
            for m in ["ETS", "ARIMA", "COMB", BAR_SIMPLE[c]]]
    return t.reindex(keep)


def check_latest(first, latest):
    check = pd.concat({"first reported": metric_table(first, "ape"),
                       "latest": metric_table(latest, "ape")}, axis=1)
    rows = pd.MultiIndex.from_product([GROUPS, ["ETS", "ARIMA", "COMB"]])
    cols = [(s, c) for c in ["h=1", "full year"] for s in ["first reported", "latest"]]
    return check.reindex(rows).loc[:, cols]


def main():
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    RESULTS.mkdir(exist_ok=True)

    first, ets_f, arima_f, comb_f = all_errors("revenue_first_reported")
    latest, _, arima_l, comb_l = all_errors("revenue_latest")
    pd.concat([arima_f.assign(series="first_reported"), arima_l.assign(series="latest")]).to_csv(
        RESULTS / "arima_forecasts.csv", index=False)
    pd.concat([comb_f.assign(series="first_reported"), comb_l.assign(series="latest")]).to_csv(
        RESULTS / "comb_forecasts.csv", index=False)

    tests = main_tests(first)
    tests.to_csv(RESULTS / "arima_main_tests.csv")
    print("MAIN TESTS (12, Holm-corrected together; 'Alphabet' with Motorola for reference only)")
    print(tests.to_string(), "\n")

    for metric, title in [("ape", "MAPE, %"), ("ase", "MASE")]:
        print(f"{title} (descriptive)")
        print(methods_table(first, metric).to_string(), "\n")
    print("ARIMA 80 % interval coverage, % of actuals inside (ETS for comparison)")
    print(pd.concat({"ARIMA": coverage(arima_f), "ETS": coverage(ets_f)}, axis=1).to_string(), "\n")
    print("ARIMA orders chosen (one per origin)")
    print(order_choices(arima_f).to_string(), "\n")
    print("MAPE, %: targets in 2020 vs other years")
    print(table_2020_methods(first).to_string(), "\n")
    print("Check: MAPE % with first-reported vs latest revenue")
    print(check_latest(first, latest).to_string(), "\n")
    print("Leak check, ARIMA (all should be 0 except the row count)")
    print(leak_check(arima_f, load()).to_string(), "\n")
    print("Disguise check: a pure simple-rule order must reproduce that rule")
    print(disguise_check(arima_f, load()).to_string())


if __name__ == "__main__":
    main()
