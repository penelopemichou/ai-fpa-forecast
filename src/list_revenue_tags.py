"""Step 3: which us-gaap tags hold each company's revenue, and for which years?

Companies changed tag names over time (new accounting rules, new
preferences), so one company's revenue can sit under several tags.
This script lists every revenue-like tag with:
  - how many values it has, and how many of those are 3-month or 12-month values
  - the first and last period it covers
Then it compares tags where they cover the same period, so we can see
which tags mean "total revenue" and which hold only a piece of it.

Run from the project folder:
    .venv/bin/python src/list_revenue_tags.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"

FILES = {
    "Mattel": "CIK0000063276.json",
    "Caterpillar": "CIK0000018230.json",
    "Google Inc.": "CIK0001288776.json",
    "Alphabet": "CIK0001652044.json",
}

# A tag is "revenue-like" if its name contains "Revenue" but none of
# these words. They mark tags that are about revenue without being
# revenue earned in a period (for example deferred revenue is a balance,
# cost of revenue is a cost).
NOT_REVENUE = ["Cost", "Deferred", "Remaining", "Liability", "ProForma",
               "EquityMethod", "Disposal", "Percentage", "Sublease", "Receivable"]


def is_revenue_like(tag):
    return "Revenue" in tag and not any(word in tag for word in NOT_REVENUE)


# --- 1. Put every revenue-like value into one table ----------------------

rows = []
for company, file_name in FILES.items():
    facts = json.loads((RAW_DIR / file_name).read_text())["facts"]["us-gaap"]
    for tag, info in facts.items():
        if not is_revenue_like(tag):
            continue
        for value in info["units"].get("USD", []):
            rows.append({"company": company, "tag": tag, **value})

df = pd.DataFrame(rows)
df["start"] = pd.to_datetime(df["start"])
df["end"] = pd.to_datetime(df["end"])
df["days"] = (df["end"] - df["start"]).dt.days

# Kind of period, by its length in days.
df["kind"] = "other"
df.loc[df["days"].between(85, 95), "kind"] = "3-month"
df.loc[df["days"].between(350, 380), "kind"] = "12-month"


# --- 2. One summary row per company and tag ------------------------------

def summarise(group):
    q = group[group["kind"] == "3-month"]
    return pd.Series({
        "values": len(group),
        "periods": group[["start", "end"]].drop_duplicates().shape[0],
        "3-month": q[["start", "end"]].drop_duplicates().shape[0],
        "12-month": group.loc[group["kind"] == "12-month", ["start", "end"]]
                         .drop_duplicates().shape[0],
        "first_end": group["end"].min().date(),
        "last_end": group["end"].max().date(),
        "filings": group["accn"].nunique(),
    })

summary = (df.groupby(["company", "tag"]).apply(summarise, include_groups=False)
             .reset_index()
             .sort_values("first_end"))

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("REVENUE-LIKE TAGS (USD values; periods = distinct start/end pairs)\n")
for company in FILES:
    print(f"== {company}")
    print(summary[summary["company"] == company].drop(columns="company")
                 .to_string(index=False))
    print()


# --- 3. Compare tags on the same 3-month periods -------------------------
# For each period we take the value from the latest filing (just for this
# comparison), then divide each tag by the company's main candidate tags.

latest = (df[df["kind"] == "3-month"]
          .sort_values("filed")
          .groupby(["company", "tag", "start", "end"])["val"].last()
          .reset_index())
wide = latest.pivot_table(index=["company", "start", "end"], columns="tag", values="val")

MAIN_TAGS = ["Revenues", "SalesRevenueNet",
             "RevenueFromContractWithCustomerExcludingAssessedTax"]

print("TAG ÷ MAIN TAG, on 3-month periods both tags cover")
print("(1.000 = same number; median, and how many periods match within 0.1%)\n")
for company in FILES:
    w = wide.loc[company].dropna(axis=1, how="all")
    lines = []
    for main in [t for t in MAIN_TAGS if t in w]:
        for other in w.columns:
            if other == main:
                continue
            both = w[[main, other]].dropna()
            if both.empty:
                continue
            ratio = both[other] / both[main]
            same = (ratio - 1).abs().le(0.001).sum()
            lines.append(f"  {other:<52} ÷ {main:<52} n={len(both):>3}  "
                         f"median {ratio.median():.3f}  same {same}/{len(both)}")
    print(f"== {company}")
    print("\n".join(lines) if lines else "  (no overlapping 3-month periods)")
    print()
