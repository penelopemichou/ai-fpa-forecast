"""Session 2, Step 3: does ETS beat the bar?

Bar = the best baseline per company in Session 1:
    Mattel B1, Caterpillar B0, Alphabet B2.

Main tests (decided before seeing any results): h = 1 and the full-year error.
For each: MAPE, % improvement over the bar, share of forecasts where ETS is
closer than the bar, and a Diebold-Mariano test with the Harvey correction.
Everything else (other horizons, MASE, coverage, model choices, 2020, the
check with latest revenue) is descriptive.

Run from the project folder:
    .venv/bin/python src/evaluate_ets.py
Saves results/ets_forecasts.csv and results/ets_main_tests.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baselines import ORIGINAL_COMPANIES, cutoff_date, known_at, load
from ets import SCALE, rolling_ets, training_series
from evaluate_baselines import (GROUPS, MOTOROLA, errors as baseline_errors, full_year,
                                metric_table, table_2020, with_groups)

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
BAR = {"Mattel": "B1", "Caterpillar": "B0", "Alphabet": "B2", "Alphabet excl. Motorola": "B2"}


# --- 1. ETS forecasts next to the baselines -------------------------------------

def all_errors(value_col, companies=ORIGINAL_COMPANIES):
    """Baselines + ETS in one long table (column 'baseline' = method name)."""
    base = baseline_errors(value_col, companies)
    ets = rolling_ets(load(value_col, companies=companies))
    scale = base.drop_duplicates(["company", "origin"])[["company", "origin", "mase_scale"]]
    e = ets.merge(scale, on=["company", "origin"])
    e["baseline"], e["forecast"] = "ETS", e["ets"]
    e["ape"] = 100 * (e["actual"] - e["forecast"]).abs() / e["actual"]
    e["ase"] = (e["actual"] - e["forecast"]).abs() / e["mase_scale"]
    e["motorola"] = (e["company"] == "Alphabet") & e["target"].between(*MOTOROLA)
    e["year_2020"] = e["target"].dt.year == 2020
    return pd.concat([base, e[base.columns]], ignore_index=True), ets


# --- 2. Diebold-Mariano test ----------------------------------------------------

def dm_test(err_model, err_bar, h):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    d = model's absolute % error - bar's absolute % error, in time order.
    If the two are equally good, d averages 0. The variance of the mean of d
    allows for correlation up to lag h-1 (forecasts h quarters ahead from
    neighbouring origins overlap). Returns (corrected statistic, two-sided p).
    Negative statistic = the model has smaller errors.
    """
    d = np.asarray(err_model) - np.asarray(err_bar)
    T, d_mean = len(d), d.mean()
    gamma = [np.sum((d[k:] - d_mean) * (d[:T - k] - d_mean)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var <= 0:                     # can happen with negative autocorrelation
        var = gamma[0] / T
    dm = d_mean / np.sqrt(var)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return hln, 2 * stats.t.sf(abs(hln), df=T - 1)


def main_tests(long):
    """h = 1 and full year: ETS vs the bar, per company group."""
    h1 = with_groups(long[long["h"] == 1])
    fy = with_groups(full_year(long))
    rows = []
    for test, data, lag in [("h=1", h1, 1), ("full year", fy, 4)]:
        for group in GROUPS:
            g = data[data["company"] == group]
            pair = g.pivot_table(index="origin", columns="baseline", values="ape").sort_index()
            ets, bar = pair["ETS"], pair[BAR[group]]
            dm, p = dm_test(ets, bar, lag)
            rows.append({"test": test, "company": group, "bar": BAR[group], "n": len(pair),
                         "MAPE ETS": ets.mean(), "MAPE bar": bar.mean(),
                         "improvement %": 100 * (bar.mean() - ets.mean()) / bar.mean(),
                         "ETS closer, % of forecasts": 100 * (ets < bar).mean(),
                         "DM (HLN)": dm, "p-value": p})
    tests = pd.DataFrame(rows).set_index(["test", "company"])
    # The family of main tests: h = 1 and full year x 3 companies (Alphabet
    # without the Motorola quarters). "Alphabet" with them is shown for reference.
    family = tests.index.get_level_values("company") != "Alphabet"
    tests.loc[family, "Holm p"] = holm(tests.loc[family, "p-value"])
    return tests


def holm(p):
    """Holm correction for running several tests: the smallest p-value is
    multiplied by the number of tests, the next by one less, and so on
    (never decreasing, never above 1)."""
    order = np.argsort(p.to_numpy())
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p.iloc[i])
        adjusted[i] = min(running_max, 1.0)
    return adjusted


# --- 3. Checks ------------------------------------------------------------------

def leak_check(ets, df):
    """Count forecasts whose training data was not public at the origin (should be 0)."""
    filed = df.set_index(["company", "quarter"])["first_filed_date"]
    train_end_filed = pd.Series([filed[(c, q)] for c, q in zip(ets["company"], ets["train_end"])],
                                index=ets.index)
    return pd.Series({
        "forecast rows": len(ets),
        "training ends after the origin": int((ets["train_end"] > ets["origin"]).sum()),
        "last training quarter filed after the cutoff":
            int((train_end_filed > cutoffs(ets, df)).sum()),
        "target not after the training data": int((ets["target"] <= ets["train_end"]).sum()),
    })


def cutoffs(rows, df):
    """The cutoff date (65 days after each company's own origin quarter ended) per row."""
    per_company = {c: g for c, g in df.groupby("company")}
    cache = {}
    out = []
    for c, o in zip(rows["company"], rows["origin"]):
        if (c, o) not in cache:
            cache[(c, o)] = cutoff_date(o, per_company[c])
        out.append(cache[(c, o)])
    return pd.Series(out, index=rows.index)


def statsmodels_check(ets, df):
    """Refit the same ETS model in statsmodels at every origin and compare h = 1 MAPE.
    If ETS's result depended on one library's optimiser, the two would differ."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    spec = {"A": "add", "M": "mul", "N": None}
    h1 = ets[ets["h"] == 1].copy()
    forecasts = []
    for _, r in h1.iterrows():
        error, trend, season = r["model"][0], r["model"][1:-1], r["model"][-1]
        known = known_at(df[df["company"] == r["company"]], r["origin"])
        y = training_series(known) / SCALE
        y.index = y.index.to_timestamp(how="end").normalize()
        fit = ETSModel(y, error=spec[error], trend=spec[trend[0]], damped_trend=trend.endswith("d"),
                       seasonal=spec[season], seasonal_periods=4).fit(disp=False, maxiter=2000)
        forecasts.append(fit.forecast((r["target"] - r["train_end"]).n).iloc[-1] * SCALE)
    h1["statsmodels"] = forecasts
    h1["ape statsforecast"] = 100 * (h1["actual"] - h1["ets"]).abs() / h1["actual"]
    h1["ape statsmodels"] = 100 * (h1["actual"] - h1["statsmodels"]).abs() / h1["actual"]
    h1["motorola"] = (h1["company"] == "Alphabet") & h1["target"].between(*MOTOROLA)
    g = with_groups(h1)
    out = g.groupby("company")[["ape statsforecast", "ape statsmodels"]].mean().reindex(GROUPS)
    out.columns = ["MAPE h=1, statsforecast", "MAPE h=1, statsmodels (same model)"]
    return out


# --- 4. Descriptive tables --------------------------------------------------------

def coverage(ets):
    """% of actuals inside the 80 % interval, by company group and horizon."""
    e = ets.assign(inside=(ets["actual"] >= ets["lo80"]) & (ets["actual"] <= ets["hi80"]),
                   motorola=(ets["company"] == "Alphabet") & ets["target"].between(*MOTOROLA))
    g = with_groups(e)
    table = 100 * g.pivot_table(index="company", columns="h", values="inside", aggfunc="mean")
    table.columns = [f"h={h}" for h in table.columns]
    table["all h"] = 100 * g.groupby("company")["inside"].mean()
    return table.reindex(GROUPS)


def model_choices(ets):
    """Which ETS models AutoETS picked (one per origin) and how often it switched."""
    per_origin = ets[ets["h"] == 1].sort_values(["company", "origin"])
    counts = per_origin.groupby("company", sort=False)["model"].value_counts().unstack(fill_value=0)
    switches = per_origin.groupby("company", sort=False)["model"].apply(
        lambda s: int((s != s.shift()).sum() - 1))
    counts["switches"] = switches
    counts["origins"] = per_origin.groupby("company", sort=False).size()
    return counts


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    RESULTS.mkdir(exist_ok=True)

    first, ets_first = all_errors("revenue_first_reported")
    latest, ets_latest = all_errors("revenue_latest")
    pd.concat([ets_first.assign(series="first_reported"),
               ets_latest.assign(series="latest")]).to_csv(RESULTS / "ets_forecasts.csv", index=False)

    tests = main_tests(first)
    tests.to_csv(RESULTS / "ets_main_tests.csv")
    print("MAIN TESTS: ETS vs the bar (first-reported revenue)")
    print(tests.to_string(), "\n")

    print("MAPE, % (all methods; descriptive)")
    print(metric_table(first, "ape").to_string(), "\n")
    print("MASE (descriptive)")
    print(metric_table(first, "ase").to_string(), "\n")
    print("80 % interval coverage, % of actuals inside")
    print(coverage(ets_first).to_string(), "\n")
    print("Models chosen by AutoETS (one per origin)")
    print(model_choices(ets_first).to_string(), "\n")
    print("MAPE, %: targets in 2020 vs other years")
    print(table_2020(first).to_string(), "\n")

    check = pd.concat({"first reported": metric_table(first, "ape"),
                       "latest": metric_table(latest, "ape")}, axis=1)
    check = check.loc[(slice(None), "ETS"),
                      [(s, c) for c in ["h=1", "full year"] for s in ["first reported", "latest"]]]
    print("Check: ETS MAPE % with first-reported vs latest revenue")
    print(check.to_string(), "\n")
    print("Leak check (all should be 0 except the row count)")
    print(leak_check(ets_first, load()).to_string(), "\n")
    print("Library check")
    print(statsmodels_check(ets_first, load()).to_string())


if __name__ == "__main__":
    main()
