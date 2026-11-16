"""Session 1, Step 2: how good are the FP&A baselines?

For every company x baseline x horizon:
  MAPE  = mean absolute percentage error: average of |actual - forecast| / actual, in %
  MASE  = mean absolute scaled error: |actual - forecast| divided by the in-sample
          error of "same quarter last year" known at the origin. Below 1 means
          better than that naive rule was on past data.
  Full year = % error on the sum of the next 4 quarters (h = 1..4 from one origin).

Alphabet targets 2012 Q2 - 2014 Q4 are affected by the Motorola restatements,
so Alphabet is reported with and without them. Targets in 2020 are shown
separately. The main run uses first-reported revenue; a second run uses the
latest revenue as a check.

Run from the project folder:
    .venv/bin/python src/evaluate_baselines.py
Saves results/baseline_forecasts.csv and results/baseline_metrics.csv.
"""

from pathlib import Path

import pandas as pd

from baselines import BASELINES, load, rolling_forecasts

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
MOTOROLA = (pd.Period("2012Q2", freq="Q"), pd.Period("2014Q4", freq="Q"))
GROUPS = ["Mattel", "Caterpillar", "Alphabet", "Alphabet excl. Motorola"]


# --- 1. Forecasts -> errors -------------------------------------------------

def errors(value_col):
    """One row per company x origin x horizon x baseline, with its errors."""
    f = rolling_forecasts(load(value_col))
    long = f.melt(id_vars=["company", "origin", "h", "target", "actual", "mase_scale"],
                  value_vars=BASELINES, var_name="baseline", value_name="forecast")
    long["ape"] = 100 * (long["actual"] - long["forecast"]).abs() / long["actual"]
    long["ase"] = (long["actual"] - long["forecast"]).abs() / long["mase_scale"]
    long["motorola"] = ((long["company"] == "Alphabet") &
                        long["target"].between(*MOTOROLA))
    long["year_2020"] = long["target"].dt.year == 2020
    return long


def with_groups(long):
    """Add the 'Alphabet excl. Motorola' group as extra rows."""
    clean = long[(long["company"] == "Alphabet") & ~long["motorola"]].assign(
        company="Alphabet excl. Motorola")
    return pd.concat([long, clean], ignore_index=True)


def full_year(long):
    """% error on the sum of the next 4 quarters, one row per origin with all 4."""
    complete = long.groupby(["company", "origin", "baseline"]).filter(lambda g: len(g) == 4)
    sums = (complete.groupby(["company", "origin", "baseline"])
                    .agg(actual=("actual", "sum"), forecast=("forecast", "sum"),
                         motorola=("motorola", "any"), year_2020=("year_2020", "any"))
                    .reset_index())
    sums["ape"] = 100 * (sums["actual"] - sums["forecast"]).abs() / sums["actual"]
    return sums


# --- 2. Tables -------------------------------------------------------------

def metric_table(long, metric):
    """Rows: company x baseline. Columns: h = 1..4, plus full year for MAPE."""
    g = with_groups(long)
    table = g.pivot_table(index=["company", "baseline"], columns="h", values=metric, aggfunc="mean")
    table.columns = [f"h={h}" for h in table.columns]
    if metric == "ape":
        fy = with_groups(full_year(long))
        table["full year"] = fy.groupby(["company", "baseline"])["ape"].mean()
    return table.reindex(GROUPS, level=0)


def table_2020(long):
    """MAPE for 2020 targets vs all other targets, h = 1 and all horizons."""
    g = with_groups(long)
    g = g[g["company"] != "Alphabet"]               # use the clean Alphabet group
    g["targets"] = g["year_2020"].map({True: "2020", False: "other years"})
    out = pd.concat({
        "h=1": g[g["h"] == 1].pivot_table(index=["company", "baseline"], columns="targets",
                                          values="ape", aggfunc="mean"),
        "all h": g.pivot_table(index=["company", "baseline"], columns="targets",
                               values="ape", aggfunc="mean"),
    }, axis=1)
    return out.reindex([x for x in GROUPS if x != "Alphabet"], level=0)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    RESULTS.mkdir(exist_ok=True)

    first = errors("revenue_first_reported")
    latest = errors("revenue_latest")
    first.assign(series="first_reported").pipe(
        lambda d: pd.concat([d, latest.assign(series="latest")])).to_csv(
        RESULTS / "baseline_forecasts.csv", index=False)

    n = first[first["baseline"] == "B1"].groupby(["company", "h"]).size().unstack()
    print("Number of forecasts (per baseline):")
    print(n.to_string(), "\n")

    mape, mase = metric_table(first, "ape"), metric_table(first, "ase")
    print("MAPE, % (first-reported revenue)")
    print(mape.to_string(), "\n")
    print("MASE (first-reported revenue; below 1 = better than 'same quarter last year' in-sample)")
    print(mase.to_string(), "\n")
    print("MAPE, %: targets in 2020 vs other years")
    print(table_2020(first).to_string(), "\n")

    check = pd.concat({"first reported": metric_table(first, "ape"),
                       "latest": metric_table(latest, "ape")}, axis=1)
    check = check.loc[:, [(s, c) for c in ["h=1", "full year"] for s in ["first reported", "latest"]]]
    print("Check: MAPE % with first-reported vs latest revenue")
    print(check.to_string())

    pd.concat({"MAPE": mape, "MASE": mase}, axis=1).to_csv(RESULTS / "baseline_metrics.csv")


if __name__ == "__main__":
    main()
