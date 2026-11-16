# ai-fpa-forecast

**Can AI forecast like an FP&A team?**

FP&A (financial planning and analysis) teams forecast next quarter's revenue every quarter, often with simple rules such as "same quarter last year, plus last year's growth". This project asks whether statistical and machine-learning models do any better, when they are tested honestly: each forecast uses only the data that was public on the day it was made.

## Companies

| Company | Why it's here | SEC CIK |
|---|---|---|
| Mattel | Toys: strong Q4 (holiday) seasonality | 0000063276 |
| Caterpillar | Machinery: follows the economic cycle | 0000018230 |
| Alphabet (incl. Google Inc. before 2015) | Fast, fairly steady growth | 0001652044, 0001288776 |

## Data source

Quarterly revenue from the SEC EDGAR XBRL API (`data.sec.gov`, companyfacts). SEC filings are public and free to reuse. Only public company data is used.

## How models are judged

- Rolling-origin forecasts from 2013 Q1, 1 to 4 quarters ahead. At each forecast date, only quarters filed with the SEC within 65 days of the forecast quarter's end are used.
- FP&A baselines every model must beat: *same quarter last year* (B1) and *same quarter last year × (1 + recent growth)* (B2), plus *last quarter* (B0) as a sanity check.
- Only first-reported numbers are used, so later restatements don't leak into the past.

## Status

**Session 0 done (16 Nov 2026): data downloaded and checked.** `data/processed/revenue_quarterly.csv` has 74 quarters per company (2008 Q2 – 2026 Q3, no gaps), with both the first-reported and the latest revenue for each quarter. See `notebooks/00_data_check.ipynb`.

**Session 1 done (16 Nov 2026): FP&A baselines.** See `notebooks/01_baselines.ipynb` and `PROGRESS.md`.

**Session 2 done (16 Nov 2026): exponential smoothing (AutoETS).** See `notebooks/02_ets.ipynb`.

**Session 3 done (16 Nov 2026): ARIMA and a simple combination.** See `notebooks/03_arima_combination.ipynb`.

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

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env              # then put your own email in .env (the SEC asks for one)
.venv/bin/python src/download_edgar.py
.venv/bin/python src/build_quarterly.py
.venv/bin/python src/evaluate_baselines.py   # Session 1: writes results/
.venv/bin/python src/evaluate_ets.py         # Session 2: writes results/
.venv/bin/python src/evaluate_arima.py       # Session 3: writes results/ (about 1 minute)
```
