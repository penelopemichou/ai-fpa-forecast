"""Session 3: one simple combination.

COMB = (ETS + the company's best simple rule) / 2, quarter by quarter, at every
horizon. The best simple rule comes from Session 1: Mattel B1, Caterpillar B0,
Alphabet B2. No weights are fitted, so nothing can leak from the future.

Run from the project folder (prints the worked example):
    .venv/bin/python src/combine.py
"""

import pandas as pd

from arima import arima_at_origin, in_disguise
from baselines import FIRST_ORIGIN, FORECASTERS, HORIZONS, cutoff_date, known_at, load, rolling_forecasts
from ets import ets_at_origin, rolling_ets

BEST_SIMPLE = {"Mattel": "B1", "Caterpillar": "B0", "Alphabet": "B2"}


def rolling_combination(df, ets=None):
    """COMB for every company x origin x horizon. Pass ETS forecasts if already made."""
    if ets is None:
        ets = rolling_ets(df)
    base = rolling_forecasts(df)
    both = ets[["company", "origin", "h", "target", "actual", "ets"]].merge(
        base[["company", "origin", "h", "B0", "B1", "B2"]], on=["company", "origin", "h"])
    both["best_simple"] = [r[BEST_SIMPLE[r["company"]]] for _, r in both.iterrows()]
    both["comb"] = (both["ets"] + both["best_simple"]) / 2
    return both[["company", "origin", "h", "target", "actual", "ets", "best_simple", "comb"]]


def worked_example(df, company, origin):
    cdf = df[df["company"] == company]
    known = known_at(cdf, origin)
    targets = [origin + h for h in HORIZONS]
    ar = arima_at_origin(known, targets)
    et = ets_at_origin(known, targets)
    best = BEST_SIMPLE[company]
    m = 1e6

    order = ar[0]["order"]
    print(f"{company}, origin {origin}: we pretend it is {cutoff_date(origin, cdf).date()}")
    print(f"ARIMA order (log revenue): {order}"
          f"{'  = ' + in_disguise(order) if in_disguise(order) else ''}")
    print(f"ETS model: {et[0]['model']}    Best simple rule: {best}\n")
    print("Forecasts ($m):")
    print(f"  {'h':<2} {'target':<7} {'ARIMA':>8}  {'80 % interval':<19} {'ETS':>8} "
          f"{'B0':>8} {'B1':>8} {'B2':>8}   COMB = (ETS + {best}) / 2")
    for h, (t, a, e) in enumerate(zip(targets, ar, et), start=1):
        b = {n: f(known, t) / m for n, f in FORECASTERS.items()}
        comb = (e["ets"] / m + b[best]) / 2
        print(f"  {h:<2} {str(t):<7} {a['arima'] / m:>8,.1f}  {a['lo80'] / m:>7,.1f} – {a['hi80'] / m:>7,.1f}   "
              f"{e['ets'] / m:>8,.1f} {b['B0']:>8,.1f} {b['B1']:>8,.1f} {b['B2']:>8,.1f}   "
              f"({e['ets'] / m:,.1f} + {b[best]:,.1f}) / 2 = {comb:,.1f}")


if __name__ == "__main__":
    worked_example(load(), "Mattel", FIRST_ORIGIN)
