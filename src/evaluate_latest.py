"""Session 6, Step 3: does starting from the latest quarter explain why ETS beat LightGBM?

Main tests (fixed before evaluating), 20 companies each, two-sided sign test:
  1. h = 1:     in how many companies does latest-start LightGBM beat ETS?
  2. full year: in how many does it beat the company's best simple rule (Session 4)?
  Companies share shocks (2020 above all), so they are not fully independent and
  the sign test is somewhat too optimistic.
Descriptive: vs Session 5 LightGBM (how much of the gap to ETS closed), vs B3 (its
simple version), the shuffled placebo, whether B3 beats the old best simple rule,
feature importance, 2020. Alphabet without the Motorola targets.

Methods in the tables:
  ETS, LGBM (Session 5, starts from the same quarter last year),
  LATEST (Session 6 LightGBM, starts from the latest quarter), PLACEBO (shuffled features),
  B0, B1, B2, B3 (latest quarter x last year's same step).

Run from the project folder:
    .venv/bin/python src/evaluate_latest.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import BASELINES, load
from companies import COMPANIES, TYPES, company_type
from evaluate_baselines import MOTOROLA, full_year
from evaluate_ets import dm_test
from evaluate_lgbm import all_methods as session5_methods, sign_test
from evaluate_types import best_rules, load_forecasts
from latest_start import CATEGORICAL, FEATURES, build_table
from lgbm import N_TREES, PARAMS
from lgbm_latest import OUT as LATEST_FORECASTS, as_model_input

ALL = list(COMPANIES)


def all_methods():
    base = session5_methods()
    g = pd.read_csv(LATEST_FORECASTS)
    for col in ["origin", "target"]:
        g[col] = pd.PeriodIndex(g[col], freq="Q")
    g = g[~((g["company"] == "Alphabet") & g["target"].between(*MOTOROLA))]
    keys = ["company", "origin", "h"]
    extra = base[base["baseline"] == "ETS"][keys + ["mase_scale", "type", "target_end", "year_2020", "motorola"]]
    parts = []
    for col, name in [("latest", "LATEST"), ("placebo", "PLACEBO"), ("b3", "B3")]:
        e = g[keys + ["target", "actual", col]].merge(extra, on=keys)
        e["baseline"], e["forecast"] = name, e[col]
        e["ape"] = 100 * (e["actual"] - e["forecast"]).abs() / e["actual"]
        e["ase"] = (e["actual"] - e["forecast"]).abs() / e["mase_scale"]
        parts.append(e[base.columns])
    return pd.concat([base] + parts, ignore_index=True)


def mape(data, method):
    return data[data["baseline"] == method].groupby("company")["ape"].mean().reindex(ALL)


def per_company(f, rules):
    h1 = f[f["h"] == 1]
    fy = full_year(f)
    t = pd.DataFrame({"type": [company_type(c) for c in ALL], "best rule": rules["best"]}, index=ALL)
    for m in ["LATEST", "ETS", "LGBM", "B3", "PLACEBO"]:
        t[f"h1 {m}"] = mape(h1, m)
    t["h1 best rule"] = [rules.loc[c, rules.loc[c, "best"]] for c in ALL]
    t["FY LATEST"] = mape(fy, "LATEST")
    t["FY rule"] = [fy[(fy["company"] == c) & (fy["baseline"] == t.loc[c, "best rule"])]["ape"].mean() for c in ALL]
    t["FY ETS"], t["FY LGBM"] = mape(fy, "ETS"), mape(fy, "LGBM")
    # DM p-values (descriptive, not corrected)
    def p(data, c, a, b, lag):
        pair = data[data["company"] == c].pivot_table(index="origin", columns="baseline", values="ape").sort_index()
        return dm_test(pair[a], pair[b], lag)[1]
    t["p vs ETS h1"] = [p(h1, c, "LATEST", "ETS", 1) for c in ALL]
    t["p vs rule FY"] = [p(fy, c, "LATEST", t.loc[c, "best rule"], 4) for c in ALL]
    t["beats ETS h1"] = t["h1 LATEST"] < t["h1 ETS"]
    t["beats rule FY"] = t["FY LATEST"] < t["FY rule"]
    t["beats B3 h1"] = t["h1 LATEST"] < t["h1 B3"]
    t["beats LGBM h1"] = t["h1 LATEST"] < t["h1 LGBM"]
    gap5, gap6 = t["h1 LGBM"] - t["h1 ETS"], t["h1 LATEST"] - t["h1 ETS"]
    t["gap to ETS, S5"], t["gap to ETS, S6"] = gap5, gap6
    t["share of gap closed"] = np.where(gap5 > 0, (gap5 - gap6) / gap5, np.nan)
    t["B3 beats old best rule"] = t["h1 B3"] < t["h1 best rule"]
    return t


def main_tests(t):
    rows = []
    for name, col in [("h=1: latest-start LightGBM beats ETS", "beats ETS h1"),
                      ("full year: latest-start LightGBM beats the best simple rule", "beats rule FY")]:
        wins = int(t[col].sum())
        rows.append({"test": name, "companies": len(t), "wins": wins, "sign test p (two-sided)": sign_test(wins, len(t))})
    return pd.DataFrame(rows).set_index("test")


def by_type(t):
    cols = ["h1 ETS", "h1 LGBM", "h1 LATEST", "h1 B3", "h1 PLACEBO", "FY LATEST", "FY rule"]
    out = t.groupby("type")[cols].mean().reindex(TYPES)
    gap5 = out["h1 LGBM"] - out["h1 ETS"]
    out["share of gap closed"] = (gap5 - (out["h1 LATEST"] - out["h1 ETS"])) / gap5
    for col in ["beats ETS h1", "beats rule FY", "beats B3 h1", "B3 beats old best rule"]:
        out[col] = t.groupby("type")[col].sum().reindex(TYPES).astype(int).astype(str) + "/5"
    return out


def b3_full_year(f, rules):
    """Descriptive: full-year MAPE of B3 vs each company's old best simple rule
    (chosen at h = 1 in Session 4) and vs ETS. Returns (per company, summary)."""
    fy = full_year(f)
    m = fy.pivot_table(index="company", columns="baseline", values="ape").reindex(ALL)
    t = pd.DataFrame({"type": [company_type(c) for c in ALL], "best rule": rules["best"]}, index=ALL)
    t["FY B3"] = m["B3"]
    t["FY best rule"] = [m.loc[c, rules.loc[c, "best"]] for c in ALL]
    t["FY ETS"] = m["ETS"]
    t["B3 beats rule"] = t["FY B3"] < t["FY best rule"]
    t["B3 beats ETS"] = t["FY B3"] < t["FY ETS"]
    rows = []
    for label, col in [("B3 beats the old best simple rule", "B3 beats rule"), ("B3 beats ETS", "B3 beats ETS")]:
        w = int(t[col].sum())
        row = {"full year": label, "wins": f"{w}/20", "sign test p (two-sided)": sign_test(w, 20)}
        row.update({ty: f"{int(t[t['type'] == ty][col].sum())}/5" for ty in TYPES})
        rows.append(row)
    return t, pd.DataFrame(rows).set_index("full year")


def all_horizons(f):
    m = f.pivot_table(index="h", columns="baseline", values="ape", aggfunc="mean")
    return m[["ETS", "LGBM", "LATEST", "B3", "B2", "PLACEBO"]]


def importance():
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


def year_2020(f):
    g = f[(f["h"] == 1) & f["baseline"].isin(["ETS", "LGBM", "LATEST"])].assign(
        when=lambda d: np.where(pd.to_datetime(d["target_end"]).dt.year == 2020, "2020", "other years"))
    return g.pivot_table(index="type", columns=["baseline", "when"], values="ape", aggfunc="mean").reindex(TYPES)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    f = all_methods()
    rules = best_rules(load_forecasts())
    t = per_company(f, rules)
    print("MAIN TESTS (sign test two-sided; companies share shocks, so p is somewhat optimistic)")
    print(main_tests(t).to_string(), "\n")
    print("Per company (MAPE %)")
    print(t.to_string(), "\n")
    print("By type")
    print(by_type(t).to_string(), "\n")
    print("Mean MAPE over all forecasts, %")
    print(all_horizons(f).to_string(), "\n")
    for col, label in [("beats LGBM h1", "latest-start beats Session 5 LightGBM at h = 1"),
                       ("beats B3 h1", "latest-start beats B3 at h = 1"),
                       ("B3 beats old best rule", "B3 beats the old best simple rule at h = 1")]:
        w = int(t[col].sum())
        print(f"{label}: {w} of 20 (sign test p = {sign_test(w, 20):.4f})")
    print(f"placebo h = 1 MAPE / real: median {np.median(t['h1 PLACEBO'] / t['h1 LATEST']):.2f}x; "
          f"placebo worse for {int((t['h1 PLACEBO'] > t['h1 LATEST']).sum())} of 20\n")
    print("Feature importance (% of split gain, final models)")
    print(importance().round(1).to_string(), "\n")
    print("h = 1 MAPE, targets ending in calendar 2020 vs other years")
    print(year_2020(f).to_string())


if __name__ == "__main__":
    main()
