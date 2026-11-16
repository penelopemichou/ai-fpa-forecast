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

- Rolling-origin forecasts: forecast one quarter ahead, move forward one quarter, repeat.
- Two FP&A baselines every model must beat: *same quarter last year*, and *same quarter last year × last year's growth*.
- Only first-reported numbers are used to train, so later restatements don't leak into the past.

## Status

**Session 0 done (16 Nov 2026): data downloaded and checked.** `data/processed/revenue_quarterly.csv` has 74 quarters per company (2008 Q2 – 2026 Q3, no gaps), with both the first-reported and the latest revenue for each quarter. See `notebooks/00_data_check.ipynb` and `PROGRESS.md`.

Next: the FP&A baselines with rolling-origin forecasts.

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env              # then put your own email in .env (the SEC asks for one)
.venv/bin/python src/download_edgar.py
.venv/bin/python src/build_quarterly.py
```
