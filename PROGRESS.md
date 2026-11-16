# Progress

## Session log

| Session | Date | What was done | Lesson |
|---|---|---|---|
| 1 | 16 Nov 2026 | **FP&A baselines.** `src/baselines.py`: B0 last quarter, B1 same quarter last year, B2 B1 × (1 + growth of the last 4 known quarters vs the 4 before), rolling origin 2013 Q1–2026 Q2, h = 1–4, 65-day availability rule (changed from 45: with 45, Mattel's Q4 was never known at Q4 origins). `src/evaluate_baselines.py`: MAPE, MASE, full-year error, 2020 separately, Alphabet with and without Motorola targets, check with latest revenue. Charts and explanations in `notebooks/01_baselines.ipynb`. Pinned NumPy 2.4.6 + pandas 2.3.3 (pandas warned with NumPy 2.5). | The best simple rule depends on the company: seasonality (Mattel → B1), growth (Alphabet → B2), cycle (Caterpillar → B0, the "sanity check", beats both FP&A rules). A model has to beat the best baseline per company, not just one fixed rule. |
| 0 | 16 Nov 2026 | **Data downloaded and checked.** Repo set up (Python 3.12 venv). Downloaded SEC companyfacts for Mattel, Caterpillar, Google Inc. and Alphabet (`src/download_edgar.py`). Listed revenue tags (`src/list_revenue_tags.py`) and chose which to stitch per company. Built `data/processed/revenue_quarterly.csv` (`src/build_quarterly.py`): 74 quarters per company, 2008 Q2–2026 Q3, no gaps, with first-reported and latest revenue. Checked it in `notebooks/00_data_check.ipynb`. | Restatements are real and can be large: Google's 2013 quarters were later restated about −7 % (Motorola). An honest forecast test must use first-reported values and the date they became public. |

## Baseline results (Session 1)

MAPE, % (first-reported revenue, 2013–2026; Alphabet without the Motorola targets 2012 Q2–2014 Q4):

| Company | B0 h=1 | B1 h=1 | B2 h=1 | B1 h=4 | B2 h=4 | Best full year |
|---|---|---|---|---|---|---|
| Mattel | 44.7 | **8.9** | 8.9 | 9.4 | 11.3 | B1 6.4 |
| Caterpillar | **7.2** | 14.1 | 12.0 | 13.9 | 19.7 | B0 9.2 |
| Alphabet | 7.5 | 15.2 | **5.7** | 15.2 | 8.8 | B2 7.2 |

- MASE at h = 1: Mattel B1 1.12 (nothing below 1); Alphabet B2 0.85; Caterpillar all below 1, mostly because 2009–2011 swings inflate its in-sample scale.
- 2020 roughly doubled the misses for Mattel and Caterpillar (Caterpillar B1 29 % vs 13 %). Alphabet's B1 did better in 2020, because growth stalled.
- First-reported vs latest revenue: the MAPE changes by less than 0.5 points; the ranking stays the same.
- My predictions: right on Mattel (B1, about 9 %) and Alphabet (B2, about 5–6 %; B1 about 15 %). Wrong on Caterpillar: I expected B2 to win, but B0 beats both FP&A rules.

## Data check results (Session 0)

- Quarters add up to the reported full year (within 0.5 %) in 50 of 51 complete years. The one failure, Alphabet 2012, is in the filings themselves: full-year 2012 was restated in 2015, the 2012 quarters never were.
- Revisions (first reported → latest): Caterpillar 0 quarters, Mattel 5 (all < 0.5 %), Alphabet 8 (up to −7.6 %, 2012–2013).
- Days until revenue was public: about a month since 2010. The 2008–2009 quarters took about 400 days, because XBRL filing started in 2009, so the first fair forecast origin is around 2010.

## Next

Session 2: the first statistical model (from *Forecasting: Principles and Practice*), judged against the best baseline per company with the same rolling origin and 65-day rule.
