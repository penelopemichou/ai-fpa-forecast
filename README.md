# ai-fpa-forecast

**Can AI forecast like an FP&A team?**

FP&A (financial planning and analysis) teams forecast next quarter's revenue every quarter, often with simple rules such as "same quarter last year, plus last year's growth". This project asks whether statistical and machine-learning models do any better, when they are tested honestly: each forecast uses only the data that was public on the day it was made.

## What I found

Short answer: next quarter, a simple smoothing model (ETS) beats the usual FP&A rules and machine learning didn't beat it; for the full year, nothing reliably beats a good simple rule.

20 large US companies, quarterly revenue, forecasts made every quarter from about 2013 to 2026 using only what was public at the time. MAPE is the average % miss, averaged over the 20 companies.

| Method | MAPE next quarter | MAPE full year | Verdict |
|---|---|---|---|
| B0 Last quarter | 12.2 % | 11.3 % | Only useful for cyclical companies |
| B1 Same quarter last year | 10.3 % | 9.4 % | The classic FP&A rule; rarely the best |
| B2 Same quarter last year × growth | 6.9 % | 7.9 % | Fine for growth companies |
| B3 Latest quarter × last year's step | 5.2 % | 7.2 % | Beats each company's old best rule next quarter for 15 of 20 |
| **B4 Latest quarter × 3-year average step** | **4.8 %** | **6.7 %** | **Best simple rule**; not reliably worse than ETS (beats it for 6 of 20) |
| **ETS** (exponential smoothing, per company) | **4.4 %** | **5.9 %** | **Best overall next quarter**; full year beats the best simple rule for only 11 of 20 |
| ARIMA, and ETS + simple rule averaged | 3 companies only | | About as good as ETS / no better; 0 of 12 tests significant |
| LightGBM, all 20 companies together | 5.1 % | 6.2 % [^lgbm] | Loses to ETS for 18 of 20 next quarter (p < 0.001) |
| LightGBM, starting from the latest quarter | 5.1 % | 6.2 % [^lgbm] | Loses to ETS for 16 of 20 (p = 0.01) |

[^lgbm]: The two LightGBM rows having the same averages is a checked coincidence, not a copy. Four quarters ahead the two models are identical by design (978 of 979 forecasts; the one exception fell back to B3 in Session 6). At one to three quarters ahead none of their 3,057 forecasts are the same, and their per-company gains and losses happen to cancel out (5.05 % vs 5.05 % next quarter; 6.22 % vs 6.16 % for the full year).

**Rule of thumb for FP&A**
- **Next quarter:** use exponential smoothing (AutoETS on the company's own history). In a spreadsheet, use B4: `=latest*GEOMEAN(three year-ago steps)`. For the 5 growth companies here, B4 was as good as ETS or better (descriptive, small sample).
- **Full year:** nothing reliably beats a good simple rule for every company. On average, ETS misses least for seasonal, cyclical and consumer-staples companies, and the latest-quarter rules (B3/B4) for growth companies.
- **Don't** default to "same quarter last year". Start from the latest quarter and smooth the seasonal step over a few years.
- Machine learning pooled across 20 companies learned real patterns but didn't beat a simple per-company model with about 1,000 training rows.

**Limitations**
- "FP&A team" here means their simple rules of thumb. Real FP&A forecasts also use inside information (order books, pricing plans, management judgment), which this project doesn't have.
- 20 large US companies that I chose, revenue only, quarterly, about 13 years including the 2020 shock.
- Revenue was stitched together from SEC XBRL tags. Q4 is often derived, and a few restatements and fiscal-year changes needed handling (see `CLAUDE.md`).
- Tests were fixed before each session, but I saw earlier sessions' results along the way: the Session 4 hypothesis came from Session 1's three companies. In Session 7 the hypothesis was fixed in chat before the run but written into PROGRESS.md only after it (a failed file edit); see PROGRESS.md.
- Each company's "best simple rule" was picked with hindsight (from its own errors, in Session 4), which flatters the simple-rule bar.
- Companies share shocks, so the sign tests are somewhat optimistic, and many comparisons were made across the sessions.
- Only MAPE: no dollar-weighted or bias measures, and prediction intervals were only checked for ETS and ARIMA.

**What I'd test next**
- **Economic drivers for cyclical companies** (for example industrial production, purchasing-manager surveys, freight volumes), using only the data vintages known at each forecast date.
- **A pretrained forecasting model** (a time-series foundation model), scored only on quarters after its training-data cutoff.
- **B4 that skips known shock years** (a post-hoc idea from Session 7), tested on new data.

## Project 3 (in progress)

Can a pretrained forecasting model (Chronos-Bolt) beat ETS and B4 on quarters published after its release? Session 1 (Mattel, 8 quarters): no real gap yet; all 20 companies next. Session 2 (20 companies, 158 quarters): Chronos-Bolt doesn’t beat ETS or B4 one quarter ahead; no test survives Holm. Session 3: the bigger base model isn’t better.

## Companies

Sessions 0–3 used three companies: Mattel, Caterpillar and Alphabet. Session 4 added 17 more, five per type, with types assigned from business descriptions only, before any results:

| Type | Companies |
|---|---|
| Seasonal (toys, retail, consumer electronics) | Mattel, Hasbro, Williams-Sonoma, Best Buy, Macy's |
| Growth tech | Alphabet (incl. Google Inc. before 2015), Microsoft, Salesforce, Netflix, Adobe |
| Cyclical industrial | Caterpillar, Deere, Cummins, PACCAR, Parker-Hannifin |
| FMCG / consumer staples | Procter & Gamble, Colgate-Palmolive, Clorox, Church & Dwight, General Mills |

CIKs and the reason for each type are in `src/companies.py`. Fiscal quarters follow each company's own fiscal year (for example, Microsoft's year ends in June).

## Data source

Quarterly revenue from the SEC EDGAR XBRL API (`data.sec.gov`, companyfacts). SEC filings are public and free to reuse. Only public company data is used.

## How models are judged

- Rolling-origin forecasts, 1 to 4 quarters ahead, starting after five years of each company's history (2013 Q1 for the original three). At each forecast date, only quarters filed with the SEC within 65 days of the forecast quarter's end are used.
- FP&A baselines every model must beat: *same quarter last year* (B1) and *same quarter last year × (1 + recent growth)* (B2), plus *last quarter* (B0) as a sanity check.
- Only first-reported numbers are used, so later restatements don't leak into the past.

## Status

**Session 0 done (16 Nov 2026): data downloaded and checked.** `data/processed/revenue_quarterly.csv` has 74 quarters per company (2008 Q2 – 2026 Q3, no gaps), with both the first-reported and the latest revenue for each quarter. See `notebooks/00_data_check.ipynb`.

**Session 1 done (16 Nov 2026): FP&A baselines.** See `notebooks/01_baselines.ipynb` and `PROGRESS.md`.

**Session 2 done (16 Nov 2026): exponential smoothing (AutoETS).** See `notebooks/02_ets.ipynb`.

**Session 3 done (16 Nov 2026): ARIMA and a simple combination.** See `notebooks/03_arima_combination.ipynb`.

**Session 4 done (17 Nov 2026): does company type decide which forecast works?** 20 companies. See `notebooks/04_company_types.ipynb`.

**Session 5 done (17 Nov 2026): machine learning (LightGBM) trained on all 20 companies.** See `notebooks/05_lightgbm.ipynb`.

**Session 6 done (17 Nov 2026): does the starting point explain why ETS beat LightGBM?** See `notebooks/06_latest_start.ipynb`.

**Session 7 done (17 Nov 2026): is smoothing the reason ETS wins?** See `notebooks/07_smoothing.ipynb`.

## Session 1 result

No single FP&A rule wins everywhere. One quarter ahead (2013–2026, first-reported revenue), the average miss (MAPE) of the best simple rule is:

| Company | Best baseline | MAPE, 1 quarter ahead | Full-year MAPE |
|---|---|---|---|
| Mattel | Same quarter last year (B1) | 8.9 % | 6.4 % |
| Caterpillar | **Last quarter (B0)** | 7.2 % | 9.2 % |
| Alphabet (excl. Motorola restatements) | Same quarter last year × growth (B2) | 5.7 % | 7.2 % |

Seasonality decides for Mattel, growth for Alphabet, and the business cycle for Caterpillar. There, both FP&A rules (12.0–14.1 %) lose to the "sanity check" of simply repeating last quarter, because year-old information is stale at cycle turns. 2020 roughly doubled the misses for Mattel and Caterpillar. Using restated instead of first-reported revenue changes the results by less than 0.5 points. These numbers are the bar for every model in later sessions.

## Session 2 result

Exponential smoothing (AutoETS, re-chosen at every origin from the quarters known then) misses less than each company's best simple rule **one quarter ahead**, but **not for the full year**:

| Company | Bar | MAPE 1 quarter ahead: ETS vs bar | Diebold–Mariano p (Holm-corrected) | Full-year MAPE: ETS vs bar |
|---|---|---|---|---|
| Mattel | B1 | 7.5 % vs 8.9 % | 0.25 (1.00) | 6.9 % vs 6.4 % |
| Caterpillar | B0 | 5.4 % vs 7.2 % | 0.02 (0.10) | 9.2 % vs 9.2 % |
| Alphabet (excl. Motorola) | B2 | 3.4 % vs 5.7 % | 0.02 (0.10) | 5.9 % vs 7.2 % |

The next-quarter gains (16–40 %) are suggestive but not conclusive once corrected for running six tests, and they fade to nothing by four quarters ahead. For the full-year number an FP&A team reports, ETS adds nothing over the simple rules. The 80 % prediction intervals held the actual 76–81 % of the time. The result survives checks for leaks, for restatements (first-reported vs latest revenue) and for the software library (statsmodels gives nearly the same errors).

## Session 3 result

AutoARIMA (on log revenue, re-chosen at every origin) and COMB (the plain average of ETS and each company's best simple rule) were tested against the current bars in 12 main tests, fixed in advance and Holm-corrected together. **None is significant (all Holm p = 1.00).**

| MAPE, % | Mattel | Caterpillar | Alphabet (excl. Motorola) |
|---|---|---|---|
| 1 quarter ahead: ETS (bar) / ARIMA / COMB | 7.5 / 6.9 / 7.3 | 5.4 / 5.9 / 6.0 | 3.4 / 3.2 / 4.2 |
| Full year: best simple rule (bar) / ARIMA / COMB | 6.4 / 6.7 / 6.4 | 9.2 / 8.1 / 9.1 | 7.2 / 5.4 / 6.1 |

ARIMA is a reasonable alternative to ETS, slightly better for some companies and slightly worse for others, and the best of all methods two to four quarters ahead for Caterpillar and Alphabet. But it switched models 15–18 times over 54 origins (ETS: 1–4). Averaging ETS with a clearly weaker simple rule did not help one quarter ahead. For the full-year number, nothing so far reliably beats the simple FP&A rules.

## Session 4 result

Hypothesis, written down before evaluating: the best simple rule one quarter ahead is *same quarter last year* (B1) for seasonal companies, *same quarter × growth* (B2) for growth tech and FMCG, and *last quarter* (B0) for cyclical industrials.

| Type | Predicted | Best simple rule (5 companies each) | Matches |
|---|---|---|---|
| Seasonal | B1 | B2 ×4, B1 ×1 (Mattel, by 0.03 points) | 1 / 5 |
| Growth tech | B2 | B2 ×5 | 5 / 5 |
| Cyclical industrial | B0 | B0 ×4, B2 ×1 (Deere, which is also seasonal) | 4 / 5 |
| FMCG | B2 | B2 ×3, B0 ×1, B1 ×1 | 3 / 5 |

**13 of 20 match**, against 8.4 on average when the type labels are shuffled at random (1.8 % of 1,000 shuffles do as well): the pre-registered test passes. But three of the matches (Mattel, Caterpillar, Alphabet) were known when the hypothesis was written. On the 17 new companies alone it is 10 of 17, against 7.5 by chance (p ≈ 0.11), so the evidence is **suggestive, not proven**. What holds up: cyclical businesses are best served by *last quarter*, and growth companies by *same quarter × growth*. Seasonal companies do *not* favour plain *same quarter last year*: they grow too, so B2 ties or beats it.

ETS beats each company's best simple rule one quarter ahead for 18 of 20 companies, but for the full year only 11 of 20, the same pattern as Sessions 2–3.

## Session 5 result

LightGBM (boosted decision trees) learned one pattern from all 20 companies together: it predicts growth vs the same quarter last year from the last four year-on-year growths, the last quarter-on-quarter growth, the fiscal quarter and the company type. It was retrained at every origin on rows public by then, with settings fixed in advance and no tuning.

| Main test (fixed in advance) | LightGBM wins | Sign test p |
|---|---|---|
| One quarter ahead: beats ETS | **2 of 20** | 0.0004 (LightGBM worse) |
| Full year: beats the company's best simple rule | 11 of 20 | 0.82 |

**Machine learning did not beat exponential smoothing.** One quarter ahead it is reliably a little worse (mean MAPE 5.0 % vs 4.4 %), and for the full year it is a coin flip against the simple rules, just like ETS. It did learn real patterns: it beats its simple version B2 for 18 of 20 companies, by learning that only part of last year's growth carries over, and a no-feature placebo for all 20. It also held up better than ETS for cyclical and FMCG companies in 2020. One quarter ahead it leans almost entirely on the latest year-on-year growth; company type matters only further out.

## Session 6 result

Session 5's LightGBM started each forecast from the same quarter last year, while ETS starts from the latest quarter. Session 6 changed only that: LightGBM now predicts the change from the latest known quarter, with the same settings, features plus "last year's same step", and a shuffled-feature placebo.

| Main test (fixed in advance) | Wins | Sign test p |
|---|---|---|
| One quarter ahead: latest-start LightGBM beats ETS | 4 of 20 | 0.012 (still worse) |
| Full year: beats the company's best simple rule | 11 of 20 | 0.82 |

**The starting point is not the whole story.** It closed the gap to ETS for growth companies (fully) and FMCG (about a third), but widened it for seasonal companies and cyclicals, so the average one-quarter-ahead error didn't move (5.04 % in both sessions). The placebo was worse for 19 of 20 companies, so the model does use its features, but it beats its own simple version B3 for only 11 of 20.

**A useful by-product: B3**, "latest quarter × last year's same step", beats each company's old best simple rule (same quarter last year, with or without growth, or last quarter) for **15 of 20 companies** (p = 0.04). It's a better default simple rule for an FP&A team, though still behind ETS (5.2 % vs 4.4 % one quarter ahead).

But B3's gain is **one quarter ahead only**. For the full year (descriptive check), it beats the old best simple rule for 12 of 20 companies (p = 0.50, a coin flip; mean 7.2 % vs 6.7 %) and ETS for only 4 of 20 (p = 0.01). The exception is growth tech: B3 beats the old rule for 5 of 5 and ETS for 4 of 5, because carrying the latest quarter forward is right when growth persists. By type, B3 beats the old rule for the full year for Seasonal 4/5, Growth 5/5, Cyclical 1/5 and FMCG 2/5.

## Session 7 result

B4 = latest quarter × the **average** of the same step over the last three years (B3 used only last year's step). In Excel: `=B21*GEOMEAN(B18/B17;B14/B13;B10/B9)`. It was fixed in advance, with no variants.

| Main test (fixed in advance) | B4 wins | Sign test p |
|---|---|---|
| One quarter ahead: B4 beats ETS | 6 of 20 | 0.12 |
| Full year: B4 beats ETS | 7 of 20 | 0.26 |

**Smoothing explains part of ETS's lead, not all of it.** B4 is the best simple rule in the project: one quarter ahead it beats B3 and each company's old best rule for 16 of 20 companies (p = 0.01), it's the best simple rule at every horizon, and unlike B3 and LightGBM it isn't reliably worse than ETS. For growth companies it beats ETS (4 of 5 next quarter, 5 of 5 for the full year). But it doesn't tie ETS overall, and seasonal companies gain nothing from it. The 2020 steps inside its 3-year average cost about 2 points on 2021–2023 forecasts for seasonal and cyclical companies. A post-hoc look suggests that leaving them out would roughly tie ETS, but that was found after seeing the results, so it's a hypothesis for new data, not a finding.

## Reproduce

```bash
python3.12 -m venv .venv
brew install libomp               # macOS only: LightGBM (Session 5) needs it
.venv/bin/pip install -r requirements.txt
cp .env.example .env              # then put your own email in .env (the SEC asks for one)
.venv/bin/python src/download_edgar.py
.venv/bin/python src/build_quarterly.py
.venv/bin/python src/evaluate_baselines.py   # Session 1: writes results/
.venv/bin/python src/evaluate_ets.py         # Session 2: writes results/
.venv/bin/python src/evaluate_arima.py       # Session 3: writes results/ (about 1 minute)
.venv/bin/python src/check_data.py           # Session 4: data checks for all 20 companies
.venv/bin/python src/evaluate_types.py run   # Session 4: forecasts for all 20 companies
.venv/bin/python src/evaluate_types.py evaluate
.venv/bin/python src/lgbm.py run             # Session 5: LightGBM forecasts (about 4 minutes)
.venv/bin/python src/evaluate_lgbm.py
.venv/bin/python src/lgbm_latest.py run      # Session 6: latest-start LightGBM + placebo (about 6 minutes)
.venv/bin/python src/evaluate_latest.py
.venv/bin/python src/evaluate_smoothing.py run        # Session 7: B4 forecasts
.venv/bin/python src/evaluate_smoothing.py evaluate
```
