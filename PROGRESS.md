# Progress

## Session log

| Session | Date | What was done | Lesson |
|---|---|---|---|
| 0 | 16 Nov 2026 | **Data downloaded and checked.** Repo set up (Python 3.12 venv). Downloaded SEC companyfacts for Mattel, Caterpillar, Google Inc. and Alphabet (`src/download_edgar.py`). Listed revenue tags (`src/list_revenue_tags.py`) and chose which to stitch per company. Built `data/processed/revenue_quarterly.csv` (`src/build_quarterly.py`): 74 quarters per company, 2008 Q2–2026 Q3, no gaps, with first-reported and latest revenue. Checked it in `notebooks/00_data_check.ipynb`. | Restatements are real and can be large: Google's 2013 quarters were later restated about −7 % (Motorola). An honest forecast test must use first-reported values and the date they became public. |

## Data check results (Session 0)

- Quarters add up to the reported full year (within 0.5 %) in 50 of 51 complete years. The one failure, Alphabet 2012, is in the filings themselves: full-year 2012 was restated in 2015, the 2012 quarters never were.
- Revisions (first reported → latest): Caterpillar 0 quarters, Mattel 5 (all < 0.5 %), Alphabet 8 (up to −7.6 %, 2012–2013).
- Days until revenue was public: about a month since 2010. The 2008–2009 quarters took about 400 days, because XBRL filing started in 2009, so the first fair forecast origin is around 2010.

## Next

Session 1: the two FP&A baselines (same quarter last year; same quarter last year × last year's growth) with rolling-origin forecasts on first-reported revenue.
