"""Session 4: does company type decide which forecast works?

Step 3 (this file's `run`): B0, B1, B2 and AutoETS for all 20 companies with the
existing code (rolling origin after 5 years of each company's history, 65-day
rule from each fiscal quarter end, first-reported revenue, h = 1-4).
Saves results/session4_forecasts.csv. No errors are looked at here.

Hypothesis (fixed before evaluating, see companies.HYPOTHESIS):
    Seasonal -> B1, Growth tech -> B2, Cyclical industrial -> B0, FMCG -> B2

Step 4 (`evaluate`):
  Main test (fixed before evaluating): for each company, the simple rule with the
  lowest h = 1 MAPE; count how many match their type's predicted rule; compare with
  1,000 random shufflings of the type labels (5 companies per type kept).
  Alphabet is scored without the Motorola targets (2012 Q2 - 2014 Q4), as in Sessions 1-3.
  Descriptive: MAPE per company and type, the gap between the best and second-best
  rule, ETS vs each company's best rule (h = 1 and full year, DM p-values listed, not
  corrected), flagged jumps, and calendar-2020 targets by type.

Run from the project folder:
    .venv/bin/python src/evaluate_types.py run        # Step 3
    .venv/bin/python src/evaluate_types.py evaluate   # Step 4
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import BASELINES, load
from companies import COMPANIES, HYPOTHESIS, TYPES, company_type
from evaluate_baselines import full_year
from evaluate_ets import all_errors, dm_test, leak_check

N_SHUFFLES = 1000
SEED = 0

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FORECASTS = RESULTS / "session4_forecasts.csv"
ALL = list(COMPANIES)


def run():
    """Forecast every company with B0, B1, B2 and ETS; save; check for leaks."""
    long, ets = all_errors("revenue_first_reported", companies=ALL)
    long["type"] = long["company"].map(company_type)
    q = load(companies=ALL)
    ends = q.set_index(["company", "quarter"])["period_end"]
    long["target_end"] = [ends[(c, t)] for c, t in zip(long["company"], long["target"])]
    RESULTS.mkdir(exist_ok=True)
    long.to_csv(FORECASTS, index=False)

    print(f"Saved {len(long):,} rows to {FORECASTS.relative_to(PROJECT)}\n")
    counts = (long[long["baseline"] == "ETS"].groupby("company", sort=False)
              .agg(type=("type", "first"), first_origin=("origin", "min"),
                   last_origin=("origin", "max"), forecasts_h1=("h", lambda h: int((h == 1).sum())),
                   forecasts_all=("h", "size")))
    print("Forecasts per company (each method has the same grid)")
    print(counts.to_string(), "\n")
    print("Leak check, ETS (all should be 0 except the row count)")
    print(leak_check(ets, q).to_string())
    print("\nHypothesis:", HYPOTHESIS)


# --- Step 4 ------------------------------------------------------------------------

def load_forecasts():
    """The saved Step 3 forecasts; Alphabet without the Motorola targets."""
    f = pd.read_csv(FORECASTS, parse_dates=["target_end"])
    for col in ["origin", "target"]:
        f[col] = pd.PeriodIndex(f[col], freq="Q")
    return f[~f["motorola"]].reset_index(drop=True)


def best_rules(f):
    """h = 1 MAPE per company and method; the best simple rule, the runner-up and the gap."""
    t = f[f["h"] == 1].pivot_table(index="company", columns="baseline", values="ape").reindex(ALL)
    ranked = t[BASELINES].apply(lambda r: r.sort_values().index.tolist(), axis=1)
    t["type"] = [company_type(c) for c in t.index]
    t["best"] = [r[0] for r in ranked]
    t["second"] = [r[1] for r in ranked]
    t["gap (points)"] = [t.loc[c, r[1]] - t.loc[c, r[0]] for c, r in zip(t.index, ranked)]
    t["predicted"] = t["type"].map(HYPOTHESIS)
    t["match"] = t["best"] == t["predicted"]
    return t[["type", "B0", "B1", "B2", "ETS", "best", "second", "gap (points)", "predicted", "match"]]


def placebo(t, n=N_SHUFFLES, seed=SEED):
    """Shuffle the type labels n times (keeping 5 per type) and count matches each time."""
    rng = np.random.default_rng(seed)
    types, best = t["type"].to_numpy(), t["best"].to_numpy()
    real = int((t["best"] == t["type"].map(HYPOTHESIS)).sum())
    counts = np.array([sum(HYPOTHESIS[ty] == b for ty, b in zip(rng.permutation(types), best))
                       for _ in range(n)])
    return real, counts


def robustness(t):
    """Two checks on the main test, descriptive only.
    1. Only the 17 new companies: the hypothesis came from Session 1's results for
       Mattel, Caterpillar and Alphabet, so those 3 matches were known in advance.
    2. Clear wins only: drop companies whose best rule wins by less than 0.5 MAPE points."""
    from baselines import ORIGINAL_COMPANIES
    new = t.drop(index=ORIGINAL_COMPANIES)
    real_new, c_new = placebo(new)
    clear = t[t["gap (points)"] >= 0.5]
    real_clear, c_clear = placebo(clear)
    return pd.DataFrame([
        {"check": "all 20 (main test)", "companies": len(t), "matches": int(t["match"].sum()),
         "shuffled mean": placebo(t)[1].mean(), "share of shuffles >= real": (placebo(t)[1] >= int(t["match"].sum())).mean()},
        {"check": "17 new companies only", "companies": len(new), "matches": real_new,
         "shuffled mean": c_new.mean(), "share of shuffles >= real": (c_new >= real_new).mean()},
        {"check": "clear wins only (gap >= 0.5 points)", "companies": len(clear), "matches": real_clear,
         "shuffled mean": c_clear.mean(), "share of shuffles >= real": (c_clear >= real_clear).mean()},
    ]).set_index("check")


def per_type(t):
    """Average h = 1 MAPE per type and method, and how often each rule was best."""
    avg = t.groupby("type")[["B0", "B1", "B2", "ETS"]].mean().reindex(TYPES)
    wins = pd.crosstab(t["type"], t["best"]).reindex(index=TYPES, columns=BASELINES, fill_value=0)
    wins.columns = [f"{c} best" for c in wins.columns]
    out = avg.join(wins)
    out["predicted"] = out.index.map(HYPOTHESIS)
    out["matches"] = t.groupby("type")["match"].sum().reindex(TYPES)
    return out


def ets_vs_best(f, t):
    """ETS vs each company's best simple rule (chosen at h = 1), at h = 1 and for the full year."""
    fy = full_year(f)
    rows = []
    for c in ALL:
        rule = t.loc[c, "best"]
        for test, data, lag in [("h=1", f[f["h"] == 1], 1), ("full year", fy, 4)]:
            pair = (data[data["company"] == c]
                    .pivot_table(index="origin", columns="baseline", values="ape").sort_index())
            dm, p = dm_test(pair["ETS"], pair[rule], lag)
            rows.append({"company": c, "type": company_type(c), "test": test, "rule": rule,
                         "MAPE ETS": pair["ETS"].mean(), "MAPE rule": pair[rule].mean(),
                         "ETS wins": pair["ETS"].mean() < pair[rule].mean(), "DM": dm, "p": p})
    return pd.DataFrame(rows)


def year_2020(f):
    """h = 1 MAPE by type: targets ending in calendar 2020 vs all other targets."""
    g = f[f["h"] == 1].assign(type=lambda d: d["company"].map(company_type),
                              when=lambda d: np.where(d["target_end"].dt.year == 2020, "2020", "other years"))
    out = g.pivot_table(index="type", columns=["baseline", "when"], values="ape", aggfunc="mean")
    out = out.reindex(TYPES)
    return out.loc[:, [(m, w) for m in ["B0", "B1", "B2", "ETS"] for w in ["2020", "other years"]]]


def evaluate():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    f = load_forecasts()
    t = best_rules(f)
    real, counts = placebo(t)

    print("MAIN TEST: does the best simple rule (h = 1 MAPE) match the type's predicted rule?")
    print(f"  real matches: {real} of {len(t)}")
    print(f"  random labellings (n = {len(counts)}): mean {counts.mean():.1f}, "
          f"range {counts.min()}-{counts.max()}; share with >= {real} matches: {(counts >= real).mean():.3f}")
    print(f"  distribution of shuffled matches: {dict(zip(*np.unique(counts, return_counts=True)))}\n")
    print("Per company (h = 1 MAPE, %)")
    print(t.to_string(), "\n")
    print("Per type")
    print(per_type(t).to_string(), "\n")
    e = ets_vs_best(f, t)
    print("ETS vs each company's best simple rule (DM p not corrected; descriptive)")
    print(e.groupby("test")["ETS wins"].agg(["sum", "size"]).to_string())
    print(e.to_string(index=False), "\n")
    print("h = 1 MAPE: targets ending in calendar 2020 vs other years, by type")
    print(year_2020(f).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    elif sys.argv[1:] == ["evaluate"]:
        evaluate()
    else:
        print(__doc__)
