# CLAUDE.md — ai-fpa-forecast

## What this repo is

"Can AI forecast like an FP&A team?" I forecast quarterly revenue of public companies from their SEC filings and compare models with the simple rules an FP&A team would use.

You (Claude) write the code. I'm a beginner in Python: explain code simply, one small step at a time.

## Rules (follow every session)

- **Public company data only (SEC EDGAR).** Never any employer data (TomTom, Kraft Heinz or any other).
- **Always compare with simple FP&A baselines**, using rolling-origin forecasts:
  - *Same quarter last year* (seasonal naive): forecast for Q3 2024 = actual Q3 2023.
  - *Same quarter last year × last year's growth*: Q3 2023 × (latest four quarters ÷ the four quarters before them).
- **No future information in training.** At each forecast date, use only data that existed then (use `revenue_first_reported` and `first_filed_date`, not restated values).
- **Run placebo/sanity checks** and say plainly when a model adds nothing over the baselines.
- **Stop at each STOP** so I can predict the result before you run anything. Give me the questions to predict.
- **Ask me before every git push.**

## Data

- Source: SEC EDGAR XBRL "companyfacts" API, `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`.
- Companies (CIK):
  - Mattel 0000063276
  - Caterpillar 0000018230
  - Alphabet 0001652044, joined with Google Inc. 0001288776 (the company before the 2015 Alphabet restructuring)
- The SEC requires a User-Agent header with a name and email. The email is read from `.env` (gitignored; see `.env.example`). Never commit it. Stay under 10 requests per second.
- `data/raw/` holds the downloaded JSON and is gitignored. `data/processed/` holds the tables we build.
- Revenue tags (in order of preference): Mattel `SalesRevenueNet` → `RevenueFromContractWithCustomerExcludingAssessedTax`; Caterpillar `Revenues`; Google Inc. `Revenues` → `SalesRevenueNet`; Alphabet Inc. `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax`.
- `data/processed/revenue_quarterly.csv`: 74 quarters per company, 2008 Q2–2026 Q3, no gaps (checked in `notebooks/00_data_check.ipynb`).
- Watch out for:
  - 2008–early 2009 quarters first appear in XBRL only as comparison figures a year later, so their `first_filed_date` is ~400 days after the quarter. Safe (conservative), but the first fair forecast origin is around 2010.
  - Alphabet 2012–2013: Google removed the Motorola businesses from past revenue, so the 2013 quarters were restated about −7 % (2012 Q2 −3.3 %). The 2015 10-K restated full-year 2012 to $46.0 bn, but the 2012 quarters never were, so they add up to the original $50.2 bn.
  - Google's original Q3 2012 10-Q has only the Jan–Sep total in the SEC data, so 2012 Q3 and Q4 count as public on 2013-10-24.
  - Q4 rule (`build_q4`): first reported = the earliest of direct Q4, full year − Q1–Q3, full year − Jan–Sep; if they disagree by more than 0.5 %, take the earliest that agrees with the newest (only Google 2012 Q4).

## Setup

- Python 3.12 (Homebrew, `/opt/homebrew/bin/python3.12`) venv in `.venv/`. Not the Mac's built-in Python 3.9, which no longer gets updates. Run things with `.venv/bin/python`, or activate first: `source .venv/bin/activate`.
- Packages: `requirements.txt` (pandas, requests, matplotlib, jupyter).

## Where I am in my learning

- Finished Andrew Ng's Machine Learning Specialization (Coursera) on 14 Nov 2026.
- Reading *Forecasting: Principles and Practice, the Pythonic Way* (otexts.com/fpppy).
- Use only methods I've covered so far. When a session needs something new, explain it simply first.

## After each session

Update `PROGRESS.md` (session log) and `README.md` (status), commit, then ask me before pushing.
