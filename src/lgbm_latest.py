"""Session 6: LightGBM "latest-start", with a shuffled placebo.

The ONE change from Session 5: start from the latest known quarter t.
    target   = log(revenue[t+h]) - log(revenue[t])     (features: src/latest_start.py)
    forecast = revenue[t] x exp(predicted change)
Everything else is as in Session 5: the same LightGBM settings and seed
(lgbm.PARAMS, lgbm.N_TREES), one model per horizon, retrained at every company
origin on all rows (any company) public by that origin's cutoff.

Placebo: at every origin, a copy of the model is trained on the same rows with
the feature rows shuffled (fixed seed), so the link between features and target
is broken. It should be clearly worse; if not, the real model learns nothing.

The forecast row is anchored at the latest known quarter t. Normally t is the
origin. If t is older (the origin quarter wasn't public yet), the model for
h' = target - t is used; if h' > 4 there is no model and B3 is used (flagged).

Run from the project folder:
    .venv/bin/python src/lgbm_latest.py           # worked example (Mattel, first origin)
    .venv/bin/python src/lgbm_latest.py run       # all -> results/session6_latest_forecasts.csv
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import HORIZONS, cutoff_date, first_origin, known_at, load
from companies import COMPANIES, TYPES
from latest_start import CATEGORICAL, FEATURES, b3_latest_times_last_years_step, build_table, row_for
from lgbm import N_TREES, PARAMS

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "session6_latest_forecasts.csv"
ALL = list(COMPANIES)
PLACEBO_SEED = 12345


def as_model_input(rows):
    X = rows[FEATURES].copy()
    X["type"] = pd.Categorical(X["type"], categories=TYPES)
    X["target_quarter"] = pd.Categorical(X["target_quarter"], categories=[1, 2, 3, 4])
    return X


class Trainer:
    """One real and one placebo model per (horizon, training set); shared across origins."""

    def __init__(self, table):
        self.table = table[table["target"].notna()].sort_values("public_date").reset_index(drop=True)
        self.cache = {}

    def models(self, h, cutoff):
        rows = self.table[(self.table["h"] == h) & (self.table["public_date"] <= cutoff)]
        key = (h, len(rows))
        if key not in self.cache:
            X, y = as_model_input(rows).reset_index(drop=True), rows["target"].to_numpy()
            real = lgb.train(PARAMS, lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL),
                             num_boost_round=N_TREES)
            rng = np.random.default_rng([PLACEBO_SEED, h, len(rows)])        # fixed, reproducible shuffle
            Xs = X.iloc[rng.permutation(len(X))].reset_index(drop=True)
            placebo = lgb.train(PARAMS, lgb.Dataset(Xs, label=y, categorical_feature=CATEGORICAL),
                                num_boost_round=N_TREES)
            self.cache[key] = (real, placebo, len(rows), rows["public_date"].max())
        return self.cache[key]


def forecast_at(trainer, cdf, company, origin, targets):
    known = known_at(cdf, origin)
    filed = cdf.set_index("quarter")["first_filed_date"]
    filed_known = filed[filed.index.isin(known.index)]
    logrev = np.log(known)
    cutoff = cutoff_date(origin, cdf)
    t = known.index[-1]                                   # the latest known quarter
    out = []
    for target in targets:
        h_model = (target - t).n
        b3 = b3_latest_times_last_years_step(known, target)
        rec = {"company": company, "origin": origin, "h": (target - origin).n, "target": target,
               "start_q": t, "start": known[t], "model_h": h_model, "b3": b3, "cutoff": cutoff}
        if h_model <= 4:
            row = row_for(logrev, filed_known, company, t, h_model)
            real, placebo, n_train, train_max = trainer.models(h_model, cutoff)
            X = as_model_input(pd.DataFrame([row]))
            change, change_p = float(real.predict(X)[0]), float(placebo.predict(X)[0])
            rec.update({"latest": known[t] * np.exp(change), "placebo": known[t] * np.exp(change_p),
                        "pred_change": change, "fallback_b3": False, "n_train": n_train,
                        "train_max_public": train_max, "row_public": row["public_date"],
                        **{f: row[f] for f in FEATURES}})
        else:                                             # no model this far ahead: use B3
            rec.update({"latest": b3, "placebo": b3, "pred_change": np.nan, "fallback_b3": True,
                        "n_train": 0, "train_max_public": pd.NaT, "row_public": pd.NaT})
        out.append(rec)
    return out


def rolling(df=None):
    if df is None:
        df = load(companies=ALL)
    trainer = Trainer(build_table(df))
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
    used = f[~f["fallback_b3"]]
    return pd.Series({
        "forecasts": len(f),
        "training data public after the cutoff": int((used["train_max_public"] > used["cutoff"]).sum()),
        "forecast row uses a quarter public after the cutoff": int((used["row_public"] > used["cutoff"]).sum()),
        "start quarter after the origin": int((f["start_q"] > f["origin"]).sum()),
        "target not after the origin": int((f["target"] <= f["origin"]).sum()),
    })


def worked_example(company="Mattel"):
    from ets import ets_at_origin
    from lgbm import Trainer as Session5Trainer, forecast_at as session5_forecast
    from features import build_table as session5_table
    df = load(companies=ALL)
    cdf = df[df["company"] == company]
    origin = first_origin(cdf)
    targets = [origin + h for h in HORIZONS]
    rows = forecast_at(Trainer(build_table(df)), cdf, company, origin, targets)
    s5 = session5_forecast(Session5Trainer(session5_table(df)), cdf, company, origin, targets)
    ets = ets_at_origin(known_at(cdf, origin), targets)
    m = 1e6
    t = rows[0]["start_q"]
    print(f"{company}, first origin {origin}: we pretend it is {rows[0]['cutoff'].date()}; "
          f"start = latest known quarter {t} = {rows[0]['start'] / m:,.1f}\n")
    print(f"  {'h':<2} {'target':<7} {'train rows':>10} {'pred. change':>13} {'latest-start':>13} "
          f"{'placebo':>9} {'B3':>9} {'Session 5':>10} {'ETS':>9}")
    for r, s, e in zip(rows, s5, ets):
        print(f"  {r['h']:<2} {str(r['target']):<7} {r['n_train']:>10,} "
              f"{100 * (np.exp(r['pred_change']) - 1):>+12.1f}% {r['latest'] / m:>13,.1f} "
              f"{r['placebo'] / m:>9,.1f} {r['b3'] / m:>9,.1f} {s['lgbm'] / m:>10,.1f} {e['ets'] / m:>9,.1f}")
    print(f"\n  latest-start = {t} x exp(predicted change); all in $m.")


def run():
    f, trainer = rolling()
    OUT.parent.mkdir(exist_ok=True)
    f.to_csv(OUT, index=False)
    print(f"Saved {len(f):,} forecasts to {OUT.relative_to(PROJECT)}; "
          f"{len(trainer.cache)} distinct training sets (each: one real + one placebo model)")
    print("Forecasts per horizon:", f.groupby("h").size().to_dict())
    print("Start quarter older than the origin:", int((f["start_q"] < f["origin"]).sum()),
          "| B3 used because no model that far ahead:", int(f["fallback_b3"].sum()))
    print("Training rows per model: min", f.loc[~f["fallback_b3"], "n_train"].min(),
          "max", f["n_train"].max(), "\n")
    print("Leak check (all should be 0 except the count)")
    print(leak_check(f).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    else:
        worked_example()
