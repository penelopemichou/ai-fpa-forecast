# Progress

## Session log

| Session | Date | What was done | Lesson |
|---|---|---|---|
| 2 | 16 Nov 2026 | **Exponential smoothing (AutoETS).** Installed statsforecast 2.1.1. `src/ets.py`: AutoETS (season length 4) re-chosen by AICc at every origin from the quarters known then, h = 1–4, 80 % intervals, chosen model saved. `src/evaluate_ets.py`: main tests fixed in advance (h = 1 and full year vs each company's bar), Diebold–Mariano with the Harvey correction, Holm correction for six tests, interval coverage, model choices, 2020, latest-revenue check, leak check, statsmodels cross-check. `notebooks/02_ets.ipynb`. Registered a project Jupyter kernel (a stale user-level `python3` kernel had been running the notebooks; reruns of 00 and 01 gave identical output). | A better next-quarter forecast is not a better full-year forecast: ETS cut the h = 1 miss by 16–40 % but tied or lost to the simple rules on the full year. With about 50 forecasts and six tests, even p ≈ 0.02 is not conclusive. |
| 1 | 16 Nov 2026 | **FP&A baselines.** `src/baselines.py`: B0 last quarter, B1 same quarter last year, B2 B1 × (1 + growth of the last 4 known quarters vs the 4 before), rolling origin 2013 Q1–2026 Q2, h = 1–4, 65-day availability rule (changed from 45: with 45, Mattel's Q4 was never known at Q4 origins). `src/evaluate_baselines.py`: MAPE, MASE, full-year error, 2020 separately, Alphabet with and without Motorola targets, check with latest revenue. Charts and explanations in `notebooks/01_baselines.ipynb`. Pinned NumPy 2.4.6 + pandas 2.3.3 (pandas warned with NumPy 2.5). | The best simple rule depends on the company: seasonality (Mattel → B1), growth (Alphabet → B2), cycle (Caterpillar → B0, the "sanity check", beats both FP&A rules). A model has to beat the best baseline per company, not just one fixed rule. |
| 0 | 16 Nov 2026 | **Data downloaded and checked.** Repo set up (Python 3.12 venv). Downloaded SEC companyfacts for Mattel, Caterpillar, Google Inc. and Alphabet (`src/download_edgar.py`). Listed revenue tags (`src/list_revenue_tags.py`) and chose which to stitch per company. Built `data/processed/revenue_quarterly.csv` (`src/build_quarterly.py`): 74 quarters per company, 2008 Q2–2026 Q3, no gaps, with first-reported and latest revenue. Checked it in `notebooks/00_data_check.ipynb`. | Restatements are real and can be large: Google's 2013 quarters were later restated about −7 % (Motorola). An honest forecast test must use first-reported values and the date they became public. |

## ETS results (Session 2)

Main tests (fixed before seeing results), first-reported revenue:

| Company | Bar | MAPE h=1 ETS / bar | DM p (Holm) | ETS closer | Full year ETS / bar | DM p (Holm) |
|---|---|---|---|---|---|---|
| Mattel | B1 | 7.5 / 8.9 | 0.25 (1.00) | 57 % | 6.9 / 6.4 | 0.68 (1.00) |
| Caterpillar | B0 | 5.4 / 7.2 | 0.02 (0.10) | 74 % | 9.2 / 9.2 | 0.87 (1.00) |
| Alphabet excl. Motorola | B2 | 3.4 / 5.7 | 0.02 (0.10) | 62 % | 5.9 / 7.2 | 0.60 (1.00) |

- The lead fades with the horizon: at h = 4 ETS ties the bar for all three companies.
- 80 % interval coverage over all horizons: Mattel 76 %, Caterpillar 79 %, Alphabet 81 %. Too narrow at h = 2 (66–81 %), too wide for Alphabet at h = 4 (94 %).
- Models: Alphabet MAM (49/54 origins), Caterpillar ANA (49/54, no trend), Mattel MNM (36) / MNA (16); only 1, 1 and 4 switches.
- 2020 at h = 1: Mattel ETS unaffected (7.8 vs 7.5 %); Caterpillar 9.5 vs 5.1 % (still beats B0); Alphabet 9.9 vs 2.8 % (slightly worse than B2).
- Checks: latest revenue changes MAPE by ≤ 0.4 points; 0 leaks in 630 forecasts; statsmodels gives nearly the same MAPE.
- My predictions: right that nothing is conclusive and that the full year shows no gain, and on Mattel's and Alphabet's models. Wrong on Caterpillar (ETS beats B0, because it chose *no* trend) and Alphabet (ETS clearly better than B2, not slightly worse), and on coverage (about right overall, not 65–70 %).

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

Session 3: to be decided. The bar is now ETS at h = 1 and the best baseline for the full year, with the same rolling origin, 65-day rule and first-reported revenue.
