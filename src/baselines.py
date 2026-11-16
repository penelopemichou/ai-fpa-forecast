"""Session 1: three simple FP&A baselines, forecast with a rolling origin.

  B0  last quarter              (sanity check only)
  B1  same quarter last year
  B2  same quarter last year x (1 + growth), where growth = the last 4
      known quarters vs the 4 quarters before them

Rolling origin: for each origin quarter (2013 Q1, 2013 Q2, ...) we pretend
today is 65 days after that quarter ended, and use only the quarters whose
first_filed_date is on or before that day. From there we forecast the next
1, 2, 3 and 4 quarters (h = 1, 2, 3, 4).

Run from the project folder (prints one worked example):
    .venv/bin/python src/baselines.py
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "processed" / "revenue_quarterly.csv"

FIRST_ORIGIN = pd.Period("2013Q1", freq="Q")
HORIZONS = [1, 2, 3, 4]
DAYS_AFTER_QUARTER_END = 65   # large companies must file the 10-K within 60 days
BASELINES = ["B0", "B1", "B2"]


# --- 1. Data ---------------------------------------------------------------

def load(value_col="revenue_first_reported"):
    """The quarterly table, with a 'quarter' column (e.g. 2013Q1) and a
    'value' column holding the revenue series we forecast."""
    df = pd.read_csv(DATA, parse_dates=["period_end", "first_filed_date"])
    df["quarter"] = pd.PeriodIndex.from_fields(
        year=df["fiscal_year"], quarter=df["fiscal_quarter"], freq="Q")
    df["value"] = df[value_col]
    return df


def cutoff_date(origin):
    """The day we pretend it is: 65 days after the origin quarter ended."""
    quarter_end = pd.Timestamp(origin.end_time.date())
    return quarter_end + timedelta(days=DAYS_AFTER_QUARTER_END)


def known_at(company_df, origin):
    """Revenue of the quarters that were public on the cutoff date,
    as a Series: quarter -> revenue, oldest first."""
    public = company_df["first_filed_date"] <= cutoff_date(origin)
    known = company_df[public & (company_df["quarter"] <= origin)]
    return known.set_index("quarter")["value"].sort_index()


# --- 2. The three baselines -------------------------------------------------
# Each one gets the known quarters and the target quarter, and returns a forecast.

def b0_last_quarter(known, target):
    """B0: next quarter(s) = the latest known quarter."""
    return known.iloc[-1]


def b1_same_quarter_last_year(known, target):
    """B1: the latest known value of the same quarter (Q1 for a Q1 target, ...).
    That is normally last year's; if last year's is not public yet, the year before."""
    same_quarter = known[known.index.quarter == target.quarter]
    return same_quarter.iloc[-1] if len(same_quarter) else np.nan


def growth_window(known):
    """The 8 most recent consecutive known quarters (oldest first), or None.
    Normally these are simply the last 8 quarters. If a quarter in between
    is not public yet, we step back to the latest complete set of 8."""
    for end in reversed(known.index):
        window = pd.period_range(end - 7, end, freq="Q")
        if all(q in known.index for q in window):
            return window
    return None


def growth(known):
    """Growth of the last 4 known quarters vs the 4 before them."""
    window = growth_window(known)
    if window is None:
        return np.nan
    return known[window[4:]].sum() / known[window[:4]].sum() - 1


def b2_same_quarter_times_growth(known, target):
    """B2: B1 x (1 + growth)."""
    return b1_same_quarter_last_year(known, target) * (1 + growth(known))


FORECASTERS = {"B0": b0_last_quarter,
               "B1": b1_same_quarter_last_year,
               "B2": b2_same_quarter_times_growth}


def seasonal_naive_scale(known):
    """MASE scale: the average in-sample error of 'same quarter last year',
    i.e. mean |y[q] - y[q - 4]| over the known quarters. Only data known
    at the origin is used."""
    errors = [abs(v - known[q - 4]) for q, v in known.items() if q - 4 in known.index]
    return np.mean(errors)


# --- 3. Rolling origin -----------------------------------------------------

def rolling_forecasts(df):
    """One row per company x origin x horizon, with every baseline's
    forecast and the actual revenue of the target quarter."""
    rows = []
    for company, cdf in df.groupby("company", sort=False):
        actual = cdf.set_index("quarter")["value"]
        last_quarter = actual.index.max()
        origin = FIRST_ORIGIN
        while origin + 1 <= last_quarter:            # last origin: h = 1 still has an actual
            known = known_at(cdf, origin)
            scale = seasonal_naive_scale(known)
            for h in HORIZONS:
                target = origin + h
                if target > last_quarter:
                    break
                row = {"company": company, "origin": origin, "h": h, "target": target,
                       "last_known": known.index[-1], "actual": actual[target],
                       "mase_scale": scale}
                for name, forecast in FORECASTERS.items():
                    row[name] = forecast(known, target)
                rows.append(row)
            origin += 1
    return pd.DataFrame(rows)


# --- 4. A worked example you can check by hand --------------------------------

def worked_example(df, company, origin):
    cdf = df[df["company"] == company]
    known = known_at(cdf, origin)
    m = 1e6                                            # show $ millions

    print(f"{company}, origin {origin}: we pretend it is {cutoff_date(origin).date()}")
    print(f"Latest public quarter: {known.index[-1]}\n")

    window = growth_window(known)
    last4, prev4 = known[window[4:]], known[window[:4]]
    print("Growth = last 4 known quarters vs the 4 before them ($m):")
    for (q_prev, v_prev), (q_last, v_last) in zip(prev4.items(), last4.items()):
        print(f"  {q_prev}  {v_prev / m:>8,.1f}      {q_last}  {v_last / m:>8,.1f}")
    g = last4.sum() / prev4.sum() - 1
    print(f"  sum     {prev4.sum() / m:>8,.1f}      sum     {last4.sum() / m:>8,.1f}"
          f"   -> growth = {last4.sum() / m:,.1f} / {prev4.sum() / m:,.1f} - 1 = {g:+.2%}\n")

    print("Forecasts ($m):")
    print(f"  {'h':<2} {'target':<7} {'B0 = last quarter':<20} {'B1 = same quarter':<24} B2 = B1 x (1 + growth)")
    for h in HORIZONS:
        target = origin + h
        base_q = known[known.index.quarter == target.quarter].index[-1]
        b0, b1, b2 = (FORECASTERS[n](known, target) / m for n in BASELINES)
        print(f"  {h:<2} {str(target):<7} {known.index[-1]} = {b0:>8,.1f}   "
              f"{base_q} = {b1:>8,.1f}      {b1:,.1f} x {1 + g:.4f} = {b2:>8,.1f}")


if __name__ == "__main__":
    worked_example(load(), "Mattel", FIRST_ORIGIN)
