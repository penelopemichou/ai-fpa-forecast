# Progress

## Session 4 hypothesis (fixed 17 Nov 2026, before any Session 4 forecast was evaluated)

Which simple rule has the lowest one-quarter-ahead MAPE, by company type (types assigned from business descriptions only, approved 16 Nov 2026):

| Type | Predicted winner | Why |
|---|---|---|
| Seasonal | B1, same quarter last year | the seasonal pattern repeats |
| Growth tech | B2, same quarter last year × growth | steady growth |
| Cyclical industrial | B0, last quarter | the cycle moves on, so year-old data is stale |
| FMCG | B2 (Penelope's prediction) | mild seasonality, steady low growth, persistent price increases 2021–24 |

Main test (fixed): count the companies whose best simple rule matches their type's prediction, then compare with 1,000 random shufflings of the type labels.

## Session 5 results: LightGBM (evaluated 17 Nov 2026)

Settings fixed in CLAUDE.md before any run (num_leaves 8, min_data_in_leaf 20, learning rate 0.05, 200 trees, seed 0; no tuning). 4,036 forecasts, 942 distinct models, 0 leaks.

| Main test (fixed) | LightGBM wins | Sign test p |
|---|---|---|
| h = 1: beats ETS | 2 of 20 (Salesforce, Adobe) | 0.0004 → LightGBM reliably worse |
| Full year: beats the best simple rule | 11 of 20 | 0.82 |

- Mean MAPE at h = 1–4: ETS 4.4 / 6.3 / 7.7 / 8.9; LightGBM 5.0 / 6.9 / 8.1 / 9.1; B2 7.0 / 8.5 / 9.6 / 10.5.
- vs B2 (its simple version) at h = 1: wins 18 of 20 (not Hasbro, P&G), median +1.3 points, up to +5.9 (cyclicals).
- Placebo (base × average training growth, no features): LightGBM better for 20 of 20 (5.1 % vs 10.3 %).
- Feature importance: h = 1 latest year-on-year growth 80 % of the gain; company type 3 % at h = 1 but 32 % at h = 4; target quarter ≈ 0.
- By type, full year vs best rule: Seasonal 4/5, Growth 4/5, Cyclical 1/5, FMCG 2/5.
- 2020 (h = 1): LightGBM better than ETS for cyclicals (14.1 vs 17.3 %) and FMCG (4.4 vs 4.9 %), worse for seasonal (16.9 vs 13.6 %).
- My predictions: vs ETS "about 6 of 20, ETS better if anything" → 2 of 20, and significantly ✓ (direction right). Full year about 11/20 ✓ (exactly 11). vs B2 about 13/20 by 0.3–0.5 points → 18/20 by about 1.3 points (bigger than expected). Importance mostly latest growth ✓ at h = 1 (but type matters at h = 4). 2020 worse than ETS for cyclicals ✗ (better). Clear failures vs ETS Caterpillar, PACCAR, Cummins, Hasbro → the biggest were Clorox, Deere, Macy's, P&G.

## Session 4 results (evaluated 17 Nov 2026)

- **Main test:** 13 of 20 match; random labellings average 8.4 (range 3–15); 1.8 % reach 13 or more → passes (p ≈ 0.02).
- **Robustness (descriptive):** the hypothesis came from Session 1's results for Mattel, Caterpillar and Alphabet. Without them: 10 of 17 vs 7.5 by chance, p ≈ 0.11. Clear wins only (gap ≥ 0.5 points): 10 of 15, p ≈ 0.07. Suggestive, not proven.
- **By type (h = 1 MAPE):** Seasonal 1/5 (B2 beat B1 for 4; Macy's and Best Buy by only 0.07 and 0.32 points); Growth 5/5 (B2); Cyclical 4/5 (B0, clear margins; Deere B2 because it is also seasonal); FMCG 3/5 (two near-ties; Colgate B0 at 1.85 %, Clorox B1).
- **Near-ties** (best vs second-best rule under 0.5 points): Mattel 0.03, Macy's 0.07, Church & Dwight 0.18, Best Buy 0.32, General Mills 0.40.
- **ETS vs each company's best rule:** h = 1 wins 18 of 20 (not PACCAR, Colgate); full year 11 of 20.
- **2020 (ETS, h = 1):** Cyclical 17.3 % vs 4.8 %, Seasonal 13.6 vs 5.6, FMCG 4.9 vs 2.8, Growth 3.8 vs 2.8.
- **Data:** 328 of 338 complete fiscal years add up within 0.5 % (the 10 failures are restatements). Eight non-seasonal jumps flagged: the 2020 shock, Hasbro–eOne, Clorox's cyberattack rebound.
- **My predictions:** 13 matches ✓ (exactly), placebo ~2 % ✓, Growth 4/5 (5), Cyclical 3/5 (4; right that Deere is seasonal), FMCG 2/5 (3), Seasonal 4/5 ✗ (1). ETS h = 1 15/20 (18), full year 8/20 (11). 2020: Cyclical and Seasonal worst ✓; least affected was Growth, not FMCG.

## Session log

| Session | Date | What was done | Lesson |
|---|---|---|---|
| 5 | 17 Nov 2026 | **LightGBM on all 20 companies.** Installed lightgbm 4.7.0 (+ libomp); native API, no scikit-learn. `src/features.py`: target = log growth vs the same quarter last year; features = last 4 year-on-year growths, last quarter-on-quarter growth, target fiscal quarter, type; each row dated by when its last piece became public. `src/lgbm.py`: one model per horizon, retrained at every company origin on all public rows; settings fixed in CLAUDE.md before running. `src/evaluate_lgbm.py`: two sign tests, per-company and per-type tables, vs B2, no-feature placebo, feature importance, 2020. `notebooks/05_lightgbm.ipynb`. | Pooling 20 companies and adding trees does not beat a simple model fitted to each company's own recent history. ML learned real patterns (beats B2 and a placebo), but starting from last year's quarter costs more than the extra data gains. |
| 4 | 17 Nov 2026 | **Company types.** 17 companies added (`src/companies.py`, five per type, types from business descriptions only). `build_quarterly.py` now uses fiscal quarters from each company's own fiscal-year ends, 13/14-week quarters, full-year values only from annual reports, and drops mis-dated full-year tags; Best Buy from fiscal 2014 (fiscal-year change). Rolling origin after 5 years of each company's history; 65-day rule from each fiscal quarter end. `src/check_data.py`: data checks incl. non-seasonal jumps and tag conflicts. `src/evaluate_types.py`: forecasts for 20 companies, main test with a label-shuffling placebo, robustness, ETS vs best rule, 2020 by type. `notebooks/04_company_types.ipynb`. Sessions 0–3 data and results reproduce byte-identically. | Company type predicts the cyclical-vs-growth split (B0 vs B2), not seasonality: seasonal companies grow too. A hypothesis built from earlier results must be tested on new companies; there the evidence was only suggestive. |
| 3 | 16 Nov 2026 | **ARIMA and a simple combination.** `src/arima.py`: AutoARIMA (season length 4) on log revenue, re-chosen by AICc at every origin, h = 1–4, 80 % intervals, order saved and flagged when it is a simple rule in disguise. `src/combine.py`: COMB = (ETS + best simple rule) / 2, no fitted weights. `src/evaluate_arima.py`: 12 main tests fixed in advance (h = 1 vs ETS, full year vs the best simple rule), Holm-corrected together; leak check; disguise check. `notebooks/03_arima_combination.ipynb`. | More flexible is not more reliable: ARIMA switched models 15–18 times and still landed within a few tenths of ETS. Averaging only helps when the two forecasts are about equally good. |
| 2 | 16 Nov 2026 | **Exponential smoothing (AutoETS).** Installed statsforecast 2.1.1. `src/ets.py`: AutoETS (season length 4) re-chosen by AICc at every origin from the quarters known then, h = 1–4, 80 % intervals, chosen model saved. `src/evaluate_ets.py`: main tests fixed in advance (h = 1 and full year vs each company's bar), Diebold–Mariano with the Harvey correction, Holm correction for six tests, interval coverage, model choices, 2020, latest-revenue check, leak check, statsmodels cross-check. `notebooks/02_ets.ipynb`. Registered a project Jupyter kernel (a stale user-level `python3` kernel had been running the notebooks; reruns of 00 and 01 gave identical output). | A better next-quarter forecast is not a better full-year forecast: ETS cut the h = 1 miss by 16–40 % but tied or lost to the simple rules on the full year. With about 50 forecasts and six tests, even p ≈ 0.02 is not conclusive. |
| 1 | 16 Nov 2026 | **FP&A baselines.** `src/baselines.py`: B0 last quarter, B1 same quarter last year, B2 B1 × (1 + growth of the last 4 known quarters vs the 4 before), rolling origin 2013 Q1–2026 Q2, h = 1–4, 65-day availability rule (changed from 45: with 45, Mattel's Q4 was never known at Q4 origins). `src/evaluate_baselines.py`: MAPE, MASE, full-year error, 2020 separately, Alphabet with and without Motorola targets, check with latest revenue. Charts and explanations in `notebooks/01_baselines.ipynb`. Pinned NumPy 2.4.6 + pandas 2.3.3 (pandas warned with NumPy 2.5). | The best simple rule depends on the company: seasonality (Mattel → B1), growth (Alphabet → B2), cycle (Caterpillar → B0, the "sanity check", beats both FP&A rules). A model has to beat the best baseline per company, not just one fixed rule. |
| 0 | 16 Nov 2026 | **Data downloaded and checked.** Repo set up (Python 3.12 venv). Downloaded SEC companyfacts for Mattel, Caterpillar, Google Inc. and Alphabet (`src/download_edgar.py`). Listed revenue tags (`src/list_revenue_tags.py`) and chose which to stitch per company. Built `data/processed/revenue_quarterly.csv` (`src/build_quarterly.py`): 74 quarters per company, 2008 Q2–2026 Q3, no gaps, with first-reported and latest revenue. Checked it in `notebooks/00_data_check.ipynb`. | Restatements are real and can be large: Google's 2013 quarters were later restated about −7 % (Motorola). An honest forecast test must use first-reported values and the date they became public. |

## ARIMA and combination results (Session 3)

Main tests (12, fixed before seeing results, Holm-corrected together), first-reported revenue:

| Test | Company | New vs | MAPE new / vs | New closer | p | Holm p |
|---|---|---|---|---|---|---|
| h = 1 | Mattel | ARIMA vs ETS | 6.9 / 7.5 | 52 % | 0.40 | 1.00 |
| | | COMB vs ETS | 7.3 / 7.5 | 57 % | 0.72 | 1.00 |
| | Caterpillar | ARIMA vs ETS | 5.9 / 5.4 | 31 % | 0.42 | 1.00 |
| | | COMB vs ETS | 6.0 / 5.4 | 31 % | 0.16 | 1.00 |
| | Alphabet excl. Motorola | ARIMA vs ETS | 3.2 / 3.4 | 60 % | 0.53 | 1.00 |
| | | COMB vs ETS | 4.2 / 3.4 | 45 % | 0.11 | 1.00 |
| Full year | Mattel | ARIMA vs B1 | 6.7 / 6.4 | 47 % | 0.76 | 1.00 |
| | | COMB vs B1 | 6.4 / 6.4 | 49 % | 0.91 | 1.00 |
| | Caterpillar | ARIMA vs B0 | 8.1 / 9.2 | 47 % | 0.38 | 1.00 |
| | | COMB vs B0 | 9.1 / 9.2 | 63 % | 0.56 | 1.00 |
| | Alphabet excl. Motorola | ARIMA vs B2 | 5.4 / 7.2 | 59 % | 0.34 | 1.00 |
| | | COMB vs B2 | 6.1 / 7.2 | 50 % | 0.39 | 1.00 |

- ARIMA has the lowest MAPE at h = 2–4 for Caterpillar and Alphabet (Alphabet h = 4: 7.7 % vs 8.8 %).
- ARIMA 80 % coverage: Mattel 75 %, Caterpillar 90 % (too wide), Alphabet 78–80 %.
- Orders: 10–12 different per company, 15 / 18 / 15 switches. Caterpillar was B0 in disguise at 11 of 54 origins (its 44 forecasts equal B0 exactly). Mattel always seasonally differenced; Alphabet seasonal difference + drift.
- 2020 at h = 1: ARIMA worse than ETS for Mattel (9.3 vs 7.8 %) and Alphabet (11.0 vs 9.9 %), better for Caterpillar (8.6 vs 9.5 %).
- Checks: 0 leaks. Restated revenue changes little, except Alphabet with Motorola quarters (ARIMA h = 1: 3.5 → 4.3 %). One fit (Alphabet, origin 2013 Q3, inside the Motorola window) used three rounds of differencing; its forecasts are sensible.
- My predictions: right that 0 of 12 are significant, right on COMB at h = 1 (worse for Alphabet ~4 % and Caterpillar ~6 %, the same for Mattel), and on the number of switches (10–20). Wrong that ARIMA's full year beats nobody (ahead for Caterpillar and Alphabet, not significant) and that its intervals are too narrow (similar or too wide).

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

Session 6: to be decided. 20 companies. The bars per company: ETS at h = 1, and the company's best simple rule (`best_rules` in `src/evaluate_types.py`) for the full year, with the same rolling origin, 65-day rule and first-reported revenue.
