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
- Companies: 20, five per type, in `src/companies.py` (CIKs, type, one-line reason). Types were assigned from business descriptions only and approved on 16 Nov 2026; they are fixed. Sessions 0–3 used only Mattel, Caterpillar and Alphabet (`ORIGINAL_COMPANIES` in `src/baselines.py`); their scripts stay pinned to those three so their results don't change.
  - Seasonal: Mattel, Hasbro, Williams-Sonoma, Best Buy, Macy's
  - Growth tech: Alphabet (joined with Google Inc. before 2015), Microsoft, Salesforce, Netflix, Adobe
  - Cyclical industrial: Caterpillar, Deere, Cummins, PACCAR, Parker-Hannifin
  - FMCG: Procter & Gamble, Colgate-Palmolive, Clorox, Church & Dwight, General Mills
  - Penelope confirmed none is a past employer.
- The SEC requires a User-Agent header with a name and email. The email is read from `.env` (gitignored; see `.env.example`). Never commit it. Stay under 10 requests per second.
- `data/raw/` holds the downloaded JSON and is gitignored. `data/processed/` holds the tables we build.
- Revenue tags (in order of preference) per company: `TAGS` in `src/build_quarterly.py`. Each list is the company's TOTAL revenue under the names it used over time; tags that were a segment or included other income in the same filing are put last or left out.
- **Fiscal quarters:** each company's own 12-month periods give its fiscal-year ends; a quarter is numbered by how many ~13-week steps it ends before that year end. Fiscal years are named after the calendar year they end in (Microsoft "2026 Q1" = Jul–Sep 2025). For December year ends this is the calendar quarter. Quarters are 80–98 days (13 or 14 weeks); `period_days` flags 14-week quarters (not adjusted).
- Data rules in `build_quarterly.py`: full-year values only from 10-K/10-K/A/8-K (not 10-Qs); a full-year value more than 25 % from the first 10-K figure is a mis-dated tag and dropped; 98 days is the longest quarter.
- `data/processed/revenue_quarterly.csv`: 20 companies, 54–77 quarters each. Checks in `src/check_data.py` (and `notebooks/00_data_check.ipynb` for the original three).
- Best Buy changed its fiscal year-end in 2012, so it is used from fiscal 2014 Q1 only (`SERIES_START` in `companies.py`).
- Watch out for:
  - 2008–early 2009 quarters first appear in XBRL only as comparison figures a year later, so their `first_filed_date` is ~400 days after the quarter. Safe (conservative), but the first fair forecast origin is around 2010.
  - Alphabet 2012–2013: Google removed the Motorola businesses from past revenue, so the 2013 quarters were restated about −7 % (2012 Q2 −3.3 %). The 2015 10-K restated full-year 2012 to $46.0 bn, but the 2012 quarters never were, so they add up to the original $50.2 bn.
  - Google's original Q3 2012 10-Q tags the quarter as starting 22 June (100 days), so it is not used; 2012 Q3 and Q4 count as public on 2013-10-24.
  - Restatements (first-reported ≠ latest by more than 0.5 %): P&G (up to 13 %), Best Buy (9 %), Microsoft (2018 revenue standard, 10 %), Alphabet (Motorola, 8 %), Clorox (6 %), Macy's, Salesforce, General Mills (≤ 1.5 %).
  - Flagged non-seasonal jumps (> 30 %): the 2020 shock (Hasbro, Macy's, Cummins, PACCAR), Hasbro–eOne (2020 Q1) and its sale (2025 Q1), Clorox's 2023 cyberattack rebound (fiscal 2024 Q2).
  - Q4 rule (`build_q4`): first reported = the earliest of direct Q4, full year − Q1–Q3, full year − Jan–Sep; if they disagree by more than 0.5 %, take the earliest that agrees with the newest (only Google 2012 Q4).

## Forecast setup (fixed for all sessions)

- **Data:** `data/processed/revenue_quarterly.csv`. The main run uses `revenue_first_reported` (inputs and actuals). A second run uses `revenue_latest` as a check.
- **Rolling origin:** the first forecast origin comes after 5 years (20 quarters) of each company's own history (for the original three: 2013 Q1), then every quarter up to the last possible one. At each origin, use only quarters whose `first_filed_date` is on or before the end of that company's own fiscal origin quarter + 65 days. Large companies must file the 10-K within 60 days (10-Q: 40 days), so an FP&A team knows the quarter by then. (Changed from 45 days on 16 Nov 2026: with 45 days, Mattel's Q4 was never known at Q4 origins.)
- **Horizons:** h = 1, 2, 3, 4 quarters ahead.
- **Baselines** (`src/baselines.py`): B0 last quarter (sanity check only), B1 same quarter last year, B2 same quarter last year × (1 + growth of the last 4 known quarters vs the 4 before them). Every model must be compared with B1 and B2.
- **Alphabet Motorola window:** targets 2012 Q2–2014 Q4 are affected by the Motorola restatements. Report Alphabet with and without them (Session 4 scores it without). Show 2020 separately (Session 4: targets ending in calendar 2020).
- **LightGBM (Session 5), fixed on 17 Nov 2026 before any run, no tuning:** one model per horizon h = 1–4, retrained at every company origin on all rows (any of the 20 companies) whose `public_date` is on or before that origin's cutoff. Parameters: `objective="regression"`, `num_leaves=8`, `min_data_in_leaf=20`, `learning_rate=0.05`, 200 trees (`num_boost_round=200`), `seed=0`; everything else default (plus `deterministic=True`, `force_row_wise=True`, `verbose=-1`, which only make runs repeatable and quiet). Target: log growth vs the same quarter last year; features in `src/features.py`. Forecast = same quarter last year × exp(predicted growth). Uses LightGBM's own API (`lgb.train`), not the scikit-learn wrapper.
- **Pre-registration:** write the hypothesis and main tests down (PROGRESS.md) before evaluating, and say when a hypothesis was shaped by earlier results on the same companies.

## Setup

- Python 3.12 (Homebrew, `/opt/homebrew/bin/python3.12`) venv in `.venv/`. Not the Mac's built-in Python 3.9, which no longer gets updates. Run things with `.venv/bin/python`, or activate first: `source .venv/bin/activate`.
- Packages: `requirements.txt` (pinned: numpy 2.4.6, pandas 2.3.3, statsforecast 2.1.1, …). Ask before installing anything new.
- **Notebooks: run them with the project kernel**, e.g. `jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=ai-fpa-forecast …`. A user-level kernel called `python3` (in `~/Library/Jupyter/kernels/`) points to a temporary venv from another project and takes priority over the venv's own. Notebooks 00 and 01 were first run on it; rerun on 16 Nov 2026 on the project kernel with identical text output.

## Where I am in my learning

- Finished Andrew Ng's Machine Learning Specialization (Coursera) on 14 Nov 2026.
- Reading *Forecasting: Principles and Practice, the Pythonic Way* (otexts.com/fpppy).
- Use only methods I've covered so far. When a session needs something new, explain it simply first.

## After each session

Update `PROGRESS.md` (session log) and `README.md` (status), commit, then ask me before pushing.
