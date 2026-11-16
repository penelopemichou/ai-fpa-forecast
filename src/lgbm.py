"""Session 5: LightGBM trained on all 20 companies, with the same rolling origin.

At every company origin (first origin after 5 years of that company's history,
65-day rule from its own fiscal quarter end):
  1. training rows = all rows of the feature table (any company) whose
     public_date is on or before this origin's cutoff date,
  2. one LightGBM model per horizon h = 1..4 (settings fixed in CLAUDE.md),
  3. the forecast row is built only from quarters known at the origin,
  4. forecast = same quarter last year x exp(predicted growth).
If last year's same quarter is not known yet (rare), the latest known same
quarter is used, as in B1, and the row is flagged.

Run from the project folder:
    .venv/bin/python src/lgbm.py           # worked example (Mattel, first origin)
    .venv/bin/python src/lgbm.py run       # all companies -> results/session5_lgbm_forecasts.csv
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import FORECASTERS, HORIZONS, cutoff_date, first_origin, known_at, load
from companies import COMPANIES, TYPES
from features import CATEGORICAL, FEATURES, build_table, row_for

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "session5_lgbm_forecasts.csv"
ALL = list(COMPANIES)

# Fixed before any run (see CLAUDE.md). No tuning.
PARAMS = {"objective": "regression", "num_leaves": 8, "min_data_in_leaf": 20,
          "learning_rate": 0.05, "seed": 0,
          "deterministic": True, "force_row_wise": True, "verbose": -1}
N_TREES = 200


def as_model_input(rows):
    """The feature columns, with the two categories typed as categories."""
    X = rows[FEATURES].copy()
    X["type"] = pd.Categorical(X["type"], categories=TYPES)
    X["target_quarter"] = pd.Categorical(X["target_quarter"], categories=[1, 2, 3, 4])
    return X


class Trainer:
    """Trains one model per (horizon, training set). The training set only grows
    with the cutoff date, so origins with the same set share one model."""

    def __init__(self, table):
        self.table = table[table["target"].notna()].sort_values("public_date").reset_index(drop=True)
        self.cache = {}

    def model(self, h, cutoff):
        rows = self.table[(self.table["h"] == h) & (self.table["public_date"] <= cutoff)]
        key = (h, len(rows))
        if key not in self.cache:
            data = lgb.Dataset(as_model_input(rows), label=rows["target"],
                               categorical_feature=CATEGORICAL, free_raw_data=False)
            booster = lgb.train(PARAMS, data, num_boost_round=N_TREES)
            self.cache[key] = (booster, len(rows), rows["public_date"].max())
        return self.cache[key]


def forecast_at(trainer, cdf, company, origin, targets):
    """LightGBM forecasts for one company and origin."""
    known = known_at(cdf, origin)
    filed = cdf.set_index("quarter")["first_filed_date"]
    filed_known = filed[filed.index.isin(known.index)]
    logrev = np.log(known)
    cutoff = cutoff_date(origin, cdf)
    out = []
    for target in targets:
        h = (target - origin).n
        row = row_for(logrev, filed_known, company, origin, h)
        booster, n_train, train_max = trainer.model(h, cutoff)
        growth = float(booster.predict(as_model_input(pd.DataFrame([row])))[0])
        base_q = target - 4
        if base_q in known.index:
            base, fallback = known[base_q], False
        else:                                           # last year's quarter not public yet: use B1's base
            base, fallback = FORECASTERS["B1"](known, target), True
        out.append({"company": company, "origin": origin, "h": h, "target": target,
                    "lgbm": base * np.exp(growth), "pred_growth": growth, "base": base,
                    "base_fallback": fallback, "cutoff": cutoff, "n_train": n_train,
                    "train_max_public": train_max, "row_public": row["public_date"],
                    **{f: row[f] for f in FEATURES}})
    return out


def rolling_lgbm(df=None, table=None):
    if df is None:
        df = load(companies=ALL)
    trainer = Trainer(build_table(df) if table is None else table)
    rows = []
    for company, cdf in df.groupby("company", sort=False):
        actual = cdf.set_index("quarter")["value"]
        last_quarter = actual.index.max()
        origin = first_origin(cdf)
        while origin + 1 <= last_quarter:
            targets = [origin + h for h in HORIZONS if origin + h in actual.index]
            for r in forecast_at(trainer, cdf, company, origin, targets):
                rows.append({**r, "actual": actual[r["target"]]})
            origin += 1
    return pd.DataFrame(rows), trainer


def leak_check(f):
    """Every forecast: training rows and the forecast row must be public by the cutoff (all 0)."""
    return pd.Series({
        "forecasts": len(f),
        "training data public after the cutoff": int((f["train_max_public"] > f["cutoff"]).sum()),
        "forecast row uses a quarter public after the cutoff": int((f["row_public"] > f["cutoff"]).sum()),
        "target not after the origin": int((f["target"] <= f["origin"]).sum()),
    })


def worked_example(company="Mattel"):
    from ets import ets_at_origin
    df = load(companies=ALL)
    cdf = df[df["company"] == company]
    origin = first_origin(cdf)
    targets = [origin + h for h in HORIZONS]
    trainer = Trainer(build_table(df))
    rows = forecast_at(trainer, cdf, company, origin, targets)
    known = known_at(cdf, origin)
    ets = ets_at_origin(known, targets)
    m = 1e6
    print(f"{company}, first origin {origin}: we pretend it is {rows[0]['cutoff'].date()}\n")
    print(f"  {'h':<2} {'target':<7} {'train rows':>10} {'pred. growth':>13}  "
          f"{'base (same q. last yr)':>23} {'LightGBM':>9} {'B2':>9} {'ETS':>9}")
    for r, e in zip(rows, ets):
        b2 = FORECASTERS["B2"](known, r["target"]) / m
        print(f"  {r['h']:<2} {str(r['target']):<7} {r['n_train']:>10,} "
              f"{100 * (np.exp(r['pred_growth']) - 1):>+12.1f}%  "
              f"{str(r['target'] - 4) + ' = ' + format(r['base'] / m, ',.1f'):>23} "
              f"{r['lgbm'] / m:>9,.1f} {b2:>9,.1f} {e['ets'] / m:>9,.1f}")
    print("\n  LightGBM = base x exp(predicted log growth); all in $m.")
    b = trainer.model(1, rows[0]["cutoff"])[0]
    imp = pd.Series(b.feature_importance("gain"), index=FEATURES)
    print("  What the h = 1 model leaned on (share of total split gain):",
          ", ".join(f"{k} {v:.0%}" for k, v in (imp / imp.sum()).sort_values(ascending=False).items()))


def run():
    f, trainer = rolling_lgbm()
    OUT.parent.mkdir(exist_ok=True)
    f.to_csv(OUT, index=False)
    print(f"Saved {len(f):,} forecasts to {OUT.relative_to(PROJECT)}; "
          f"{len(trainer.cache)} distinct models trained")
    print("Forecasts per horizon:", f.groupby("h").size().to_dict())
    print("Forecasts whose base is not last year's quarter (B1 fallback):", int(f["base_fallback"].sum()))
    print("Training rows per model: min", f["n_train"].min(), "max", f["n_train"].max(), "\n")
    print("Leak check (all should be 0 except the count)")
    print(leak_check(f).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    else:
        worked_example()
