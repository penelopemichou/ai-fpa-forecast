"""Session 3: AutoARIMA on log revenue, with the same rolling origin as the baselines.

At every origin:
  1. take the quarters known at that origin (65-day rule, src/baselines.py),
  2. take logs, let AutoARIMA choose the best ARIMA model by AICc,
  3. forecast the next 1-4 quarters with an 80 % interval, then convert back (exp).

Why logs: revenue grows and swings in %, not in fixed dollar amounts. On the log
scale a 10 % change is the same size whether revenue is $1 bn or $100 bn, so a
straight-line model of log revenue is a constant-% model of revenue (ETS handled
this with its multiplicative "M" letters). Converting back with exp gives the
median forecast, and exp of the interval ends gives the 80 % interval.

Run from the project folder (prints the chosen order at one origin):
    .venv/bin/python src/arima.py
"""

import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA

from baselines import FIRST_ORIGIN, HORIZONS, first_origin, known_at, load
from ets import LEVEL, SEASON_LENGTH, training_series


def order_name(fitted):
    """'(p,d,q)(P,D,Q)4', plus '+ drift' or '+ mean' if the model has a constant."""
    p, q, P, Q, m, d, D = fitted.model_["arma"]
    name = f"({p},{d},{q})({P},{D},{Q}){m}"
    coef = fitted.model_["coef"]
    if "drift" in coef:
        name += " + drift"
    elif "intercept" in coef:
        name += " + mean"
    return name


def in_disguise(order):
    """Plain-words name when the chosen ARIMA is just a simple rule (on log revenue)."""
    return {
        "(0,1,0)(0,0,0)4": "B0: last quarter",
        "(0,1,0)(0,0,0)4 + drift": "last quarter x average quarterly growth",
        "(0,0,0)(0,1,0)4": "B1: same quarter last year",
        "(0,0,0)(0,1,0)4 + drift": "same quarter last year x average yearly growth (like B2)",
        "(0,1,0)(0,1,0)4": "last quarter + last year's change from that quarter to the next",
    }.get(order, "")


def arima_at_origin(known, targets):
    """Fit AutoARIMA on log revenue of the known quarters and forecast each target."""
    if not targets:
        return []
    train = training_series(known)
    steps = [(t - train.index[-1]).n for t in targets]
    model = AutoARIMA(season_length=SEASON_LENGTH).fit(np.log(train.to_numpy()))
    pred = model.predict(h=max(steps), level=[LEVEL])
    mean = np.asarray(pred["mean"])
    lo, hi = np.asarray(pred[f"lo-{LEVEL}"]), np.asarray(pred[f"hi-{LEVEL}"])
    order = order_name(model)
    return [{"target": t, "arima": np.exp(mean[s - 1]),
             "lo80": np.exp(lo[s - 1]), "hi80": np.exp(hi[s - 1]),
             "order": order, "aicc": model.model_["aicc"],
             "train_start": train.index[0], "train_end": train.index[-1],
             "n_train": len(train)} for t, s in zip(targets, steps)]


def rolling_arima(df):
    """One row per company x origin x horizon (same grid as the baselines)."""
    rows = []
    for company, cdf in df.groupby("company", sort=False):
        actual = cdf.set_index("quarter")["value"]
        last_quarter = actual.index.max()
        origin = first_origin(cdf)                   # after 5 years of history
        while origin + 1 <= last_quarter:
            known = known_at(cdf, origin)
            targets = [origin + h for h in HORIZONS if origin + h in actual.index]
            for row in arima_at_origin(known, targets):
                rows.append({"company": company, "origin": origin,
                             "h": (row["target"] - origin).n,
                             "actual": actual[row["target"]], **row})
            origin += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    known = known_at(df[df["company"] == "Mattel"], FIRST_ORIGIN)
    r = arima_at_origin(known, [FIRST_ORIGIN + 1])[0]
    print(f"Mattel, origin {FIRST_ORIGIN}: ARIMA order {r['order']}"
          f"{'  = ' + in_disguise(r['order']) if in_disguise(r['order']) else ''}")
