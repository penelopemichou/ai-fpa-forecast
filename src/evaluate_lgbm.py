"""Session 5, Step 4: does LightGBM (trained on all 20 companies) beat the bars?

Main tests (fixed before evaluating), 20 companies each:
  1. h = 1:     in how many companies does LightGBM have a lower MAPE than ETS?
  2. full year: in how many companies does LightGBM beat the company's best simple
                rule (chosen at h = 1 in Session 4)?
  Each with a two-sided sign test against 50/50. Note: companies share shocks
  (2020 above all), so they are not fully independent and the test is somewhat
  too optimistic.
Descriptive: per-company MAPE and DM p-values (not corrected), results by type,
LightGBM vs B2 (its simple version), feature importance, 2020.
Alphabet is scored without the Motorola targets, as in Sessions 1-4.

Run from the project folder:
    .venv/bin/python src/evaluate_lgbm.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from baselines import load
from companies import COMPANIES, TYPES, company_type
from evaluate_baselines import MOTOROLA, full_year
from evaluate_ets import dm_test
from evaluate_types import best_rules, load_forecasts
from features import CATEGORICAL, FEATURES, build_table
from lgbm import N_TREES, OUT as LGBM_FORECASTS, PARAMS, as_model_input

ALL = list(COMPANIES)
METHODS = ["B0", "B1", "B2", "ETS", "LGBM"]


def all_methods():
    """Session 4's B0/B1/B2/ETS forecasts plus LightGBM, one long table, Alphabet without Motorola."""
    base = load_forecasts()
    g = pd.read_csv(LGBM_FORECASTS, parse_dates=["cutoff"])
    for col in ["origin", "target"]:
        g[col] = pd.PeriodIndex(g[col], freq="Q")
    g["motorola"] = (g["company"] == "Alphabet") & g["target"].between(*MOTOROLA)
    g = g[~g["motorola"]].drop(columns="type")         # 'type' comes back from the merge below
    keys = ["company", "origin", "h"]
    g = g.merge(base[base["baseline"] == "ETS"][keys + ["mase_scale", "type", "target_end", "year_2020"]], on=keys)
    g["baseline"], g["forecast"] = "LGBM", g["lgbm"]
    g["ape"] = 100 * (g["actual"] - g["forecast"]).abs() / g["actual"]
    g["ase"] = (g["actual"] - g["forecast"]).abs() / g["mase_scale"]
    return pd.concat([base, g[base.columns]], ignore_index=True)


def sign_test(wins, n):
    return stats.binomtest(wins, n, 0.5, alternative="two-sided").pvalue


def paired(data, company, a, b, lag):
    pair = data[data["company"] == company].pivot_table(index="origin", columns="baseline", values="ape").sort_index()
    dm, p = dm_test(pair[a], pair[b], lag)
    return pair[a].mean(), pair[b].mean(), p


def per_company(f, rules):
    h1 = f[f["h"] == 1]
    fy = full_year(f)
    rows = []
    for c in ALL:
        rule = rules.loc[c, "best"]
        lg, et, p_ets = paired(h1, c, "LGBM", "ETS", 1)
        _, b2, p_b2 = paired(h1, c, "LGBM", "B2", 1)
        lg_fy, rule_fy, p_fy = paired(fy, c, "LGBM", rule, 4)
        ets_fy = fy[(fy["company"] == c) & (fy["baseline"] == "ETS")]["ape"].mean()
        rows.append({"company": c, "type": company_type(c), "best rule": rule,
                     "h1 LGBM": lg, "h1 ETS": et, "p vs ETS": p_ets, "h1 B2": b2, "p vs B2": p_b2,
                     "FY LGBM": lg_fy, "FY rule": rule_fy, "p FY vs rule": p_fy, "FY ETS": ets_fy})
    t = pd.DataFrame(rows).set_index("company")
    t["beats ETS h1"] = t["h1 LGBM"] < t["h1 ETS"]
    t["beats rule FY"] = t["FY LGBM"] < t["FY rule"]
    t["beats B2 h1"] = t["h1 LGBM"] < t["h1 B2"]
    return t


def main_tests(t):
    rows = []
    for name, col in [("h=1: LightGBM beats ETS", "beats ETS h1"),
                      ("full year: LightGBM beats the best simple rule", "beats rule FY")]:
        wins, n = int(t[col].sum()), len(t)
        rows.append({"test": name, "companies": n, "LightGBM wins": wins, "sign test p (two-sided)": sign_test(wins, n)})
    return pd.DataFrame(rows).set_index("test")


def by_type(t):
    out = t.groupby("type")[["h1 LGBM", "h1 ETS", "h1 B2", "FY LGBM", "FY rule", "FY ETS"]].mean().reindex(TYPES)
    for col in ["beats ETS h1", "beats B2 h1", "beats rule FY"]:
        out[col] = t.groupby("type")[col].sum().reindex(TYPES).astype(int).astype(str) + "/5"
    return out


def all_horizons(f):
    m = f.pivot_table(index="h", columns="baseline", values="ape", aggfunc="mean")[METHODS]
    m.columns.name = "mean MAPE over all forecasts, %"
    return m


def importance(table=None):
    """Share of split gain per feature in the final model of each horizon (trained on every public row)."""
    if table is None:
        table = build_table(load(companies=ALL))
    table = table[table["target"].notna()]
    out = {}
    for h in [1, 2, 3, 4]:
        rows = table[table["h"] == h]
        b = lgb.train(PARAMS, lgb.Dataset(as_model_input(rows), label=rows["target"],
                                          categorical_feature=CATEGORICAL), num_boost_round=N_TREES)
        gain = pd.Series(b.feature_importance("gain"), index=FEATURES)
        out[f"h={h}"] = 100 * gain / gain.sum()
    return pd.DataFrame(out).sort_values("h=1", ascending=False)


def average_growth_placebo(f, table=None):
    """Sanity check: the same forecast without any features. At each origin,
    forecast = base x exp(average target of all training rows public by the cutoff).
    If LightGBM can't beat this, its features add nothing. Returns h = 1 MAPE per company."""
    if table is None:
        table = build_table(load(companies=ALL))
    table = table[table["target"].notna()].sort_values("public_date")
    g = f[(f["baseline"] == "LGBM") & (f["h"] == 1)][["company", "origin", "actual", "ape"]].merge(
        pd.read_csv(LGBM_FORECASTS, parse_dates=["cutoff"]).assign(
            origin=lambda d: pd.PeriodIndex(d["origin"], freq="Q"))
        .query("h == 1")[["company", "origin", "cutoff", "base"]], on=["company", "origin"])
    t1 = table[table["h"] == 1]
    dates, cum = t1["public_date"].to_numpy(), np.cumsum(t1["target"].to_numpy())
    idx = np.searchsorted(dates, g["cutoff"].to_numpy(), side="right")
    g["placebo"] = g["base"] * np.exp(cum[idx - 1] / idx)
    g["ape_placebo"] = 100 * (g["actual"] - g["placebo"]).abs() / g["actual"]
    out = g.groupby("company")[["ape", "ape_placebo"]].mean().reindex(ALL)
    out.columns = ["h1 LGBM", "h1 average-growth placebo"]
    out["LGBM better"] = out["h1 LGBM"] < out["h1 average-growth placebo"]
    return out


def year_2020(f):
    g = f[(f["h"] == 1) & f["baseline"].isin(["B2", "ETS", "LGBM"])].assign(
        when=lambda d: np.where(pd.to_datetime(d["target_end"]).dt.year == 2020, "2020", "other years"))
    return g.pivot_table(index="type", columns=["baseline", "when"], values="ape", aggfunc="mean").reindex(TYPES)


def main():
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    f = all_methods()
    rules = best_rules(load_forecasts())
    t = per_company(f, rules)
    print("MAIN TESTS (20 companies; sign test two-sided; companies share shocks, so p is somewhat optimistic)")
    print(main_tests(t).to_string(), "\n")
    print("Per company: MAPE %, DM p-values not corrected (descriptive)")
    print(t.to_string(), "\n")
    print("By type (mean MAPE %, and wins out of 5)")
    print(by_type(t).to_string(), "\n")
    w = int(t["beats B2 h1"].sum())
    gap = (t["h1 B2"] - t["h1 LGBM"])
    print(f"LightGBM vs B2 at h = 1: wins {w} of 20, sign test p = {sign_test(w, 20):.3f}; "
          f"median gain {gap.median():.2f} points (range {gap.min():.2f} to {gap.max():.2f})\n")
    print(all_horizons(f).to_string(), "\n")
    print("Feature importance (% of split gain, final models)")
    print(importance().round(1).to_string(), "\n")
    print("h = 1 MAPE, targets ending in calendar 2020 vs other years")
    print(year_2020(f).to_string())


if __name__ == "__main__":
    main()
