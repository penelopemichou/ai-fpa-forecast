# Progress

## Project 3 – Session 3 (base model)

19 Nov 2026. One change from Session 2: `amazon/chronos-bolt-base` (205M parameters, revision `5d9f166`; repo created 25 Nov 2024, the same day as small) instead of small (48M). Same 158 quarters, median, one quarter ahead, 65-day rule, same history as ETS. `src/chronos_base.py`; tables `results/chronos_base_*.csv`. Checks: same quarters as small, same history as ETS, identical on a rerun, small-model files untouched.

- **MAPE (average of the companies' MAPEs):** base 5.36 % vs small 5.27 %, ETS 4.29 %, B4 4.23 %.
- **Wins:** base beats small for 10 of 20 companies, ETS for 6 of 20, B4 for 7 of 20.
- **By type:** base is better than small on seasonal (4 of 5; 6.11 vs 6.69 %) and FMCG (4 of 5; 4.36 vs 4.80 %), worse on all 5 growth tech (3.37 vs 2.23 %). It pushes growth further (lean 1.63 vs 1.41); Netflix was too high in all 8 quarters (+3.6 % on average; MAPE 3.56 vs small 1.04 %).
- **Seasonal lean** (0 = same quarter last year, 1 = last quarter): base 0.13, small 0.23, ETS −0.06.
- **Range:** 82 % of actuals inside base's 10–90 % range.
- **Tests (Holm within each family):** sign tests vs ETS p = 0.12, vs B4 p = 0.26, vs small p = 1.00; 60 per-company Diebold–Mariano tests, 14 with raw p < 0.05, all against base; smallest Holm p = 0.20. **No test survives Holm.**
- **Conclusion:** the bigger base model isn't better.

**Claude's prediction vs the results** (`notes/predictions.md`):

| Prediction | Result | |
|---|---|---|
| Base 5.0 %, a little better than small's 5.27 %, still behind ETS and B4 | 5.36 %: a little worse than small, behind ETS and B4 | ✗ (direction) |
| Base beats ETS for 9 of 20, B4 for 9 of 20 | 6 and 7 of 20 | ✗ |
| Deere still the worst company, about 15 % | Worst, 16.35 % | ✓ |
| Seasonal lean still about 0.2 | 0.13 (down from small's 0.23) | ≈ |
| About 85 % inside the 10–90 % range | 82 % | ✓ |
| No test survives Holm (vs ETS, B4 and small) | None survives | ✓ |

## Project 3 – Session 2 (all 20 companies)

19 Nov 2026. Same setup as Session 1 (chronos-bolt-small, median, CPU, one quarter ahead, 65-day rule, same history as ETS), on all 20 companies. `src/chronos_all.py`; tables `results/chronos_all_*.csv`.

- **Quarters:** actual first published after 25 Nov 2024, checked per company with its own filing dates → 158 quarters (7–8 per company). Checks: Chronos saw exactly ETS's quarters at every origin; the ETS refit equals the Project 2 ETS forecasts; Session 1's Mattel file reproduces byte-identically.
- **MAPE (average of the companies' MAPEs):** Chronos 5.27 %, ETS 4.29 %, B4 4.23 %. Chronos beats ETS for 8 of 20 companies and B4 for 9 of 20.
- **Still last without Deere** (Chronos 17.8 % vs ETS 5.5 %, 0 of 8 quarters won): Chronos 4.61 %, ETS 4.22 %, B4 4.11 % for the other 19.
- **Seasonal companies:** Chronos leans towards "last quarter". On a scale from 0 (same quarter last year) to 1 (last quarter), its median is 0.23 vs ETS −0.06, so it misses seasonal dips (Macy's after the holiday quarter).
- **Range:** 85 % of actuals inside Chronos's 10–90 % range (135 of 158; vs 80 %, p ≈ 0.09), at most slightly too wide. ETS's 80 % range: 79 %.
- **Tests (Holm within each family):** sign tests across companies, vs ETS p = 0.50 and vs B4 p = 0.82; 40 per-company Diebold–Mariano tests, 6 with raw p < 0.05 (all in favour of ETS or B4), smallest Holm p = 0.24. **No test survives Holm.**
- **Clorox gap:** fiscal 2026 Q4 (Apr–Jun 2026) is missing from `revenue_quarterly.csv`, so Clorox has 7 quarters and its fiscal 2027 Q1 was forecast two steps ahead (by Chronos and ETS alike). Cause: Clorox filed its fiscal 2026 10-K on 7 Aug 2026 (accession 0000021076-26-000034, with XBRL), but the SEC companyfacts file downloaded on 16 Nov 2026 contains no facts from that filing. Clorox reports no separate Q4 figure, so Q4 = full year − Q1–Q3 cannot be built without the full-year number. Data and results left unchanged; a fresh download may fix it.
- **Conclusion:** Chronos-Bolt doesn't beat ETS or B4 one quarter ahead; no test survives Holm.

**Claude's prediction vs the results** (`notes/predictions.md`):

| Prediction | Result | |
|---|---|---|
| MAPE Chronos 5.8, ETS 5.6, B4 6.2 % (roughly tied) | 5.27 / 4.29 / 4.23 %: Chronos last | ✗ |
| Chronos beats ETS 9 of 20, B4 11 of 20 | 8 and 9 of 20 | ≈ |
| ~90 % inside the 80 % range (too wide) | 85 % (p ≈ 0.09), at most slightly wide | ≈ |
| Chronos too low in 13 of 20 | 9 of 20; average error +0.08 % | ✗ |
| Best on seasonal and FMCG, worst on growth tech (too low) | Closest to ETS on growth tech (2.23 vs 2.03 %, average error −0.12 %); weakest on cyclicals (Deere) and FMCG (1 of 5 wins) | ✗ |
| No test survives Holm | None survives | ✓ |
| Mattel's win was one lucky quarter | Mattel still a win, but it doesn't carry over to the other companies | ✓ |

## Project 3 – Session 1 (Mattel)

19 Nov 2026. A pretrained forecasting model, scored only where it cannot have seen the answer.

- **Model:** `amazon/chronos-bolt-small` (48M parameters, revision `772f3d2`), zero-shot (not trained on Mattel), median forecast, CPU. `src/chronos_bolt.py`.
- **Rule:** the model card states no release date; the Hugging Face repo was created on 25 Nov 2024 (first commit 13 Nov). Scored quarters: actual first published after 25 Nov 2024 → 8 quarters, 2024 Q4 to 2026 Q3, one quarter ahead. Same 65-day rule and same history as ETS (checked: identical quarters at every origin).
- **MAPE on the same 8 quarters:** Chronos 5.15 %, ETS 6.16 %, B4 6.60 %. Chronos closer than ETS in 6 of 8 (sign test p = 0.29), than B4 in 5 of 8 (p = 0.73).
- **2025 Q4 drives the lead:** ETS missed by 13.0 % and B4 by 17.0 %, Chronos by 6.2 %. Without that quarter: Chronos 5.01, B4 5.11, ETS 5.18 %.
- **Range:** 8 of 8 actuals inside Chronos's 10–90 % range (about 6–7 expected), so the range is possibly too wide (8 of 8 inside happens ~17 % of the time by chance even with an honest range; check in Session 2). It is 15–20 % of the forecast for Q3–Q4 and 31–46 % for Q1–Q2.
- **Claude's prediction** (`notes/predictions.md`): Chronos 8 %, ETS 7 %, B4 7.5 %; 3 and 4 wins of 8; 6 of 8 inside → Chronos did better than predicted, and the range was wider than predicted.
- **Conclusion:** no real gap on 8 quarters; needs all 20 companies.

## Session 7 hypothesis (fixed 17 Nov 2026 in Penelope's message, before any B4 forecast was computed)

**Note on timing:** the hypothesis and predictions below were fixed in the chat before anything ran. My edit putting them into this file failed unnoticed, so this entry was written *after* the evaluation ran. The text is unchanged from what was fixed beforehand.

**Rule (fixed, no variants):** B4 = latest known quarter × the geometric mean of the same step over the last 3 years, revenue[t+h−4k] / revenue[t−4k] for k = 1, 2, 3 (fewer if fewer are public). `src/smoothing.py`.

**Hypothesis:** if smoothing explains ETS's lead, B4 ties ETS (about 10 of 20 companies) and beats B3 mostly for seasonal and cyclical companies.

**Main tests (fixed):** (1) one quarter ahead, in how many of the 20 does B4 beat ETS? (2) full year, in how many of the 20 does B4 beat ETS? Two-sided sign tests; companies share shocks, so they aren't fully independent.

**Penelope's predictions:** h = 1 about 8 of 20 (closer than B3's 4, not a tie: B4 smooths the step but still starts from one quarter, while ETS also smooths the level); full year about 7 of 20; B4 beats B3 for about 14 of 20 at h = 1 (mostly seasonal and FMCG), about 13 of 20 for the full year; B4 beats the old best rule for about 15 of 20 at h = 1, about 12 of 20 for the full year; growth tech about the same as B3; 2020 steps in the 3-year average make 2021–2023 forecasts worse by about 1–2 points for seasonal and cyclical companies, barely for growth and FMCG. Reading if B4 lands near 8 of 20: smoothing the step explains part of ETS's lead; the rest is smoothing the starting level.

## Session 7 results: B4, smoothing the step (evaluated 17 Nov 2026)

4,036 B4 forecasts (4,032 with 3 years of steps, 4 with 2), 0 leaks.

| Main test (fixed) | B4 wins | Sign test p |
|---|---|---|
| h = 1: B4 beats ETS | 6 of 20 (Alphabet, Salesforce, Netflix, Adobe, Parker-Hannifin, P&G) | 0.12 |
| Full year: B4 beats ETS | 7 of 20 | 0.26 |

- Hypothesis (B4 ties ETS at about 10 of 20, beats B3 mostly for seasonal and cyclical): not confirmed. B4 is not reliably worse than ETS (unlike B3 and LightGBM), but 6 is short of a tie; it beats B3 for cyclicals 5/5 and FMCG 5/5 but seasonal only 2/5.
- h = 1: B4 beats B3 16/20 (p = 0.01) and the old best rule 16/20 (p = 0.01). Full year: vs B3 14/20 (p = 0.12), vs the old best rule 10/20 (p = 1.00).
- Mean MAPE h = 1–4: B4 4.8 / 6.8 / 8.4 / 9.9 (best simple rule at every horizon); ETS 4.4 / 6.3 / 7.7 / 8.9.
- Growth tech: B4 beats ETS 4/5 at h = 1 (2.68 vs 2.88 %) and 5/5 for the full year (4.31 vs 5.43 %). Seasonal: no gain from smoothing (6.86 vs B3 6.80 %).
- 2020 steps inside the 3-year average (h = 1, 2021–2023 forecasts): cost Seasonal 2.1, Cyclical 1.9, Growth 0.3, FMCG 0.1 points. Post-hoc (after seeing results, not a finding): without them B4 would average 4.51 % vs ETS 4.41 % and beat ETS for 10 of 20.
- My predictions: h = 1 about 8 → 6; full year about 7 ✓ (exactly 7); vs B3 about 14 at h = 1 → 16 (cyclical and FMCG, not seasonal); full year about 13 → 14; vs old best rule about 15 → 16 ✓ and about 12 for the full year → 10; growth about the same as B3 (roughly ✓, B4 slightly better next quarter); 2020 cost 1–2 points for seasonal and cyclical, barely for growth and FMCG ✓ (2.1, 1.9, 0.3, 0.1).

## Session 4 hypothesis (fixed 17 Nov 2026, before any Session 4 forecast was evaluated)

Which simple rule has the lowest one-quarter-ahead MAPE, by company type (types assigned from business descriptions only, approved 16 Nov 2026):

| Type | Predicted winner | Why |
|---|---|---|
| Seasonal | B1, same quarter last year | the seasonal pattern repeats |
| Growth tech | B2, same quarter last year × growth | steady growth |
| Cyclical industrial | B0, last quarter | the cycle moves on, so year-old data is stale |
| FMCG | B2 (Penelope's prediction) | mild seasonality, steady low growth, persistent price increases 2021–24 |

Main test (fixed): count the companies whose best simple rule matches their type's prediction, then compare with 1,000 random shufflings of the type labels.

## Session 6 results: latest-start LightGBM (evaluated 17 Nov 2026)

One change from Session 5: start from the latest known quarter (target = log change from it; new feature = last year's same step). Same LightGBM settings and seed. 4,036 forecasts, 943 training sets (each a real and a shuffled-placebo model), 0 leaks. Fixed during the run: an integer overflow in B3 for large companies (whole-number revenues multiplied before dividing).

| Main test (fixed) | Wins | Sign test p |
|---|---|---|
| h = 1: latest-start LightGBM beats ETS | 4 of 20 (Salesforce, Netflix, Adobe, Church & Dwight) | 0.012 → still reliably worse |
| Full year: beats the best simple rule | 11 of 20 | 0.82 |

- vs Session 5 at h = 1: better for 14 of 20 (p = 0.12), but improvements (−3.6 points) and setbacks (+3.6) cancel; mean MAPE 5.04 % in both. Gap to ETS by type: Growth 0.08 → −0.06 (closed), FMCG 1.06 → 0.72, Seasonal 0.48 → 0.74, Cyclical 0.86 → 1.11 (Deere, PACCAR worse; Caterpillar 83 % closed).
- vs B3 (its simple version): 11 of 20 (p = 0.82). The ML adds almost nothing on top of B3.
- **B3 beats the old best simple rule for 15 of 20 (p = 0.04):** all growth, 4/5 seasonal (not Hasbro), 3/5 cyclical (not Cummins, PACCAR), 3/5 FMCG (not Colgate, Clorox). Mean h = 1 MAPE: B3 5.2 %, B2 7.0 %, ETS 4.4 %.
- B3 full year (descriptive, added after the session): beats the old best simple rule for 12 of 20 (p = 0.50; mean 7.2 vs 6.7 %) and ETS for 4 of 20 (p = 0.01). By type vs the old rule: Seasonal 4/5, Growth 5/5, Cyclical 1/5, FMCG 2/5; vs ETS: Growth 4/5, others 0/5. B3's gain is one quarter ahead only, except for growth tech.
- Placebo (shuffled features): worse for 19 of 20, median 2.3× (seasonal 4.6×). Exception PACCAR (8.6 vs 10.0 %): a shuffled model ≈ "latest × average change", close to PACCAR's best rule B0.
- Feature importance: last year's same step 88 % (h = 1) and 79 % (h = 2); latest quarter-on-quarter change 74 % at h = 3; h = 4 is the Session 5 model.
- 2020 (h = 1) vs Session 5: better for seasonal (16.0 vs 16.9 %) and growth (3.8 vs 4.3 %), worse for cyclical (16.4 vs 14.1 %) and FMCG (4.6 vs 4.4 %).
- My predictions: vs ETS about 8 of 20, not significant → 4 of 20, significant (the gap didn't close as much as I thought). Full year about 11 ✓ (exactly 11). Gap half to most closed, most for cyclicals → on average none; closed for growth and FMCG, widened for cyclicals and seasonal. vs B3 about 15 of 20 → 11. Placebo 2–3× worse, worst for seasonal ✓ (2.3×, seasonal 4.6×). B3 beats the old best rule for about 5 → 15, including Hasbro and Clorox losing ✓. Last year's step dominates at h = 1 ✓ (88 %); qoq second only at h = 3. 2020 about the same as Session 5 overall ✓ (mixed by type).

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
| Wrap-up | 17 Nov 2026 | README "What I found" at the top: one table of all methods (mean MAPE over the 20 companies, next quarter and full year), a rule of thumb for FP&A, limitations and what I'd test next. Numbers recomputed one consistent way (average of company MAPEs). | Next quarter: ETS beats the usual FP&A rules, and machine learning didn't beat it. Full year: nothing reliably beats a good simple rule. The best simple rule found is B4 (latest quarter × 3-year average step). |
| 7 | 17 Nov 2026 | **Smoothing: B4.** `src/smoothing.py`: B4 = latest quarter × geometric mean of the same step over 3 years (Excel version in the notebook). `src/evaluate_smoothing.py`: B4 on the full grid with a leak check, two sign tests vs ETS, vs B3 and the old best rule, by type, the cost of 2020 steps, and a clearly labelled post-hoc diagnostic. `notebooks/07_smoothing.ipynb`. The PROGRESS.md entry for the hypothesis was written after the run (a failed edit); the hypothesis itself was fixed in the chat beforehand. | Averaging three years of steps gives the best simple rule so far, but still not ETS. A tempting post-hoc fix (drop the 2020 steps) must wait for new data. |
| 6 | 17 Nov 2026 | **Latest-start LightGBM.** `src/latest_start.py`: B3 (latest quarter × last year's same step) and the new target/feature. `src/lgbm_latest.py`: same LightGBM settings, starting from the latest known quarter, plus a shuffled-feature placebo at every origin. `src/evaluate_latest.py`: two sign tests, gap closed vs Session 5, vs B3, B3 vs the old simple rules, placebo, importance, 2020. `notebooks/06_latest_start.ipynb`. | Changing one thing isolates its effect: the starting point mattered for some types and hurt others, and wasn't why ETS won. The clearest gain came from the simple rule, not the model: B3 beats the old best simple rule for 15 of 20. |
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

Project wrapped up on 17 Nov 2026; see README "What I found". Ideas for later are listed there under "What I'd test next". The bars per company: ETS at h = 1, and the company's best simple rule (`best_rules` in `src/evaluate_types.py`) for the full year, with the same rolling origin, 65-day rule and first-reported revenue.
