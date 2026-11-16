"""Step 4: build one quarterly revenue table -> data/processed/revenue_quarterly.csv

What it does, in order:
  1. Read the revenue values from the raw SEC files, using the tags we chose
     in Step 3 (in order of preference).
  2. Keep only 3-month values (one quarter) and 12-month values (full year).
     Year-to-date values (6 or 9 months) are dropped.
  3. Each quarter is usually reported in several filings. We keep:
       revenue_first_reported = value in the earliest filing (what an analyst knew then)
       revenue_latest         = value in the latest filing (after any restatements)
  4. Q4 is often not reported on its own, or only reported much later.
     Then Q4 = full year - Q1 - Q2 - Q3, or full year - the Jan-Sep total
     (the same sum; the only use of a year-to-date value). The first-reported
     Q4 is whichever way was public first; q4_derived = True when that was a
     derived value. See build_q4 for the details.
  5. Google Inc. and Alphabet are joined into one company, "Alphabet",
     after checking that the quarters both of them report agree.

Run from the project folder:
    .venv/bin/python src/build_quarterly.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
OUT_FILE = PROJECT / "data" / "processed" / "revenue_quarterly.csv"

# Company -> list of (raw file, entity name, tags in order of preference).
# Alphabet has two entries: Google Inc. (until 2015) and Alphabet Inc.
COMPANIES = {
    "Mattel": [
        ("CIK0000063276.json", "Mattel",
         ["SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"]),
    ],
    "Caterpillar": [
        ("CIK0000018230.json", "Caterpillar", ["Revenues"]),
    ],
    "Alphabet": [
        ("CIK0001288776.json", "Google Inc.", ["Revenues", "SalesRevenueNet"]),
        # In 2019-2024 Alphabet tagged many quarters only with the second tag.
        # Where both exist they are the same number (14 of 14 in Step 3).
        ("CIK0001652044.json", "Alphabet Inc.",
         ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]),
    ],
}

QUARTER_DAYS = (85, 95)   # a 3-month period is 85-95 days long
NINE_MONTH_DAYS = (265, 285)  # Jan-Sep year-to-date; only used to derive Q4
YEAR_DAYS = (350, 380)    # a 12-month period is 350-380 days long


# --- 1. Read the raw values ---------------------------------------------

def load_facts():
    """One row per (tag, period, filing) for the tags we chose."""
    rows = []
    for company, sources in COMPANIES.items():
        for file_name, entity, tags in sources:
            facts = json.loads((RAW_DIR / file_name).read_text())["facts"]["us-gaap"]
            for priority, tag in enumerate(tags):
                for v in facts.get(tag, {}).get("units", {}).get("USD", []):
                    if "start" not in v:      # skip values without a period
                        continue
                    rows.append({"company": company, "entity": entity, "tag": tag,
                                 "priority": priority, "start": v["start"], "end": v["end"],
                                 "val": v["val"], "accn": v["accn"], "form": v["form"],
                                 "filed": v["filed"]})
    df = pd.DataFrame(rows)
    for col in ["start", "end", "filed"]:
        df[col] = pd.to_datetime(df[col])
    return df


def prepare(df):
    """Keep quarters and full years, one value per period per filing."""
    days = (df["end"] - df["start"]).dt.days
    df["kind"] = None
    df.loc[days.between(*QUARTER_DAYS), "kind"] = "quarter"
    df.loc[days.between(*NINE_MONTH_DAYS), "kind"] = "jan-sep"
    df.loc[days.between(*YEAR_DAYS), "kind"] = "year"
    df = df[df["kind"].notna()].copy()     # drops 6-month year-to-date values

    # All three companies have fiscal years that end on 31 December, so
    # the period's end date tells us the fiscal year and quarter.
    # A quarter must end in March, June, September or December; a full
    # year in December; Jan-Sep in September. Anything else is reported
    # and dropped.
    month = df["end"].dt.month
    good = (((df["kind"] == "quarter") & month.isin([3, 6, 9, 12])) |
            ((df["kind"] == "year") & (month == 12)) |
            ((df["kind"] == "jan-sep") & (month == 9)))
    odd = df[~good]
    if len(odd):
        print(f"Dropped {len(odd)} values with an unexpected end date:")
        print(odd[["entity", "tag", "start", "end", "kind"]].to_string(index=False))
    df = df[good].copy()
    df["fiscal_year"] = df["end"].dt.year
    df["fiscal_quarter"] = df["end"].dt.month // 3        # 3->1, 6->2, 9->3, 12->4

    # If one filing has the same period under two of our tags, keep the
    # preferred tag (lowest priority number).
    df = (df.sort_values("priority")
            .drop_duplicates(["entity", "kind", "fiscal_year", "fiscal_quarter", "accn"])
            .sort_values(["filed", "accn"]))
    return df


# --- 2. One row per quarter ------------------------------------------------

def as_of(filings, date):
    """The latest value among filings made on or before `date`."""
    return filings[filings["filed"] <= date].iloc[-1]


def build_q4(quarters, years, jan_sep, fy):
    """One Q4 row. There are up to three ways to get Q4:
         direct:  a 3-month Q4 value in a filing
         derived: full year - Q1 - Q2 - Q3
         derived: full year - Jan-Sep total (same sum; used for 2008-2009,
                  where Q1 is missing or became public late)
    """
    direct = quarters[(quarters["fiscal_year"] == fy) & (quarters["fiscal_quarter"] == 4)]
    year = years[years["fiscal_year"] == fy]

    # The pieces that add up to Jan-Sep, in two versions.
    options = []
    q123 = [quarters[(quarters["fiscal_year"] == fy) & (quarters["fiscal_quarter"] == q)]
            for q in (1, 2, 3)]
    if all(len(p) for p in q123):
        options.append(q123)
    if len(jan_sep[jan_sep["fiscal_year"] == fy]):
        options.append([jan_sep[jan_sep["fiscal_year"] == fy]])
    if year.empty:
        options = []

    # Each way gives a value and the day it became known.
    candidates = []   # (date known, value, derived?, tag)
    if len(direct):
        candidates.append((direct["filed"].iloc[0], direct["val"].iloc[0], False,
                           direct["tag"].iloc[0]))
    for parts in options:
        date = max([year["filed"].min()] + [p["filed"].min() for p in parts])
        year_then = as_of(year, date)
        value = year_then["val"] - sum(as_of(p, date)["val"] for p in parts)
        candidates.append((date, value, True, year_then["tag"]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])

    # First reported = the earliest way. But if the ways disagree, an early
    # one mixed old and restated pieces (Google 2012: Motorola Home was
    # removed from the full year before Jan-Sep was restated). Then take
    # the earliest way that agrees (within 0.5 %) with the newest one.
    newest = candidates[-1][1]
    date, first, derived, tag = next(c for c in candidates if abs(c[1] / newest - 1) <= 0.005)

    # Latest = the latest direct Q4 value; otherwise full year - latest pieces.
    # The full year must be on the same basis as the pieces, so we take it
    # from the first filing on or after the pieces were last restated.
    # (Google 2013: quarters restated in 2014 10-Qs, full year in the 2015
    # 10-K. Google 2012: full year restated in 2015, but the 2012 quarters
    # never were, so the 2015 full year is not used.)
    if len(direct):
        latest, period_end = direct["val"].iloc[-1], direct["end"].iloc[-1]
    else:
        parts = options[0]
        pieces_last_filed = max(p["filed"].max() for p in parts)
        later = year[year["filed"] >= pieces_last_filed]
        year_same_basis = later.iloc[0] if len(later) else year.iloc[-1]
        latest = year_same_basis["val"] - sum(p["val"].iloc[-1] for p in parts)
        period_end = year["end"].iloc[-1]

    return {"period_end": period_end.date(), "revenue_first_reported": first,
            "revenue_latest": latest, "first_filed_date": date.date(),
            "source_tag": tag, "q4_derived": derived}


def build_table(df, name):
    """Quarterly table for one company (or one entity) from its filings."""
    quarters = df[df["kind"] == "quarter"]
    years = df[df["kind"] == "year"]
    jan_sep = df[df["kind"] == "jan-sep"]
    rows = []
    for fy in sorted(set(quarters["fiscal_year"]) | set(years["fiscal_year"])):
        for fq in (1, 2, 3, 4):
            filings = quarters[(quarters["fiscal_year"] == fy) &
                               (quarters["fiscal_quarter"] == fq)]
            if fq == 4:                           # direct or derived
                row = build_q4(quarters, years, jan_sep, fy)
            elif len(filings):                    # reported as a 3-month value
                first, last = filings.iloc[0], filings.iloc[-1]
                row = {"period_end": last["end"].date(),
                       "revenue_first_reported": first["val"],
                       "revenue_latest": last["val"],
                       "first_filed_date": first["filed"].date(),
                       "source_tag": first["tag"], "q4_derived": False}
            else:
                row = None
            if row:
                rows.append({"company": name, "fiscal_year": fy, "fiscal_quarter": fq, **row})
    return pd.DataFrame(rows)


# --- 3. Check Google Inc. against Alphabet Inc. --------------------------

def check_google_alphabet(df):
    """Compare the quarters that both Google Inc. and Alphabet Inc. report."""
    a = df[df["company"] == "Alphabet"]
    google = build_table(a[a["entity"] == "Google Inc."], "Google Inc.")
    alphabet = build_table(a[a["entity"] == "Alphabet Inc."], "Alphabet Inc.")
    both = google.merge(alphabet, on=["fiscal_year", "fiscal_quarter"],
                        suffixes=("_google", "_alphabet"))
    print("\nQuarters reported by both Google Inc. and Alphabet Inc.:")
    if both.empty:
        print("  none")
        return
    both["diff_latest_%"] = 100 * (both["revenue_latest_alphabet"] /
                                   both["revenue_latest_google"] - 1)
    cols = ["fiscal_year", "fiscal_quarter", "revenue_latest_google",
            "revenue_latest_alphabet", "diff_latest_%",
            "first_filed_date_google", "first_filed_date_alphabet"]
    print(both[cols].to_string(index=False))


# --- 4. Run everything -----------------------------------------------------

def main():
    df = prepare(load_facts())
    check_google_alphabet(df)

    # Joined table: Google Inc. and Alphabet Inc. filings are pooled, so
    # "first reported" is the earliest filing of either one.
    table = pd.concat([build_table(df[df["company"] == c], c) for c in COMPANIES],
                      ignore_index=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {len(table)} rows to {OUT_FILE.relative_to(PROJECT)}\n")
    for company, t in table.groupby("company", sort=False):
        print(f"{company:<12} {len(t):>3} quarters, "
              f"{t.iloc[0]['fiscal_year']} Q{t.iloc[0]['fiscal_quarter']} to "
              f"{t.iloc[-1]['fiscal_year']} Q{t.iloc[-1]['fiscal_quarter']}, "
              f"Q4 derived {t['q4_derived'].sum()} times")


if __name__ == "__main__":
    main()
