# Predictions

Written before each run, so I can check them afterwards.

## Project 3 – Session 1 (Mattel)

Saved 19 Nov 2026, before any Chronos-Bolt forecast for Mattel was run.

Claude’s prediction (Mattel, 1 quarter ahead, 2024 Q4–2026 Q3)

* MAPE: Chronos 8 %, ETS 7 %, B4 7.5 %
* Chronos beats ETS in 3 of 8 quarters and beats B4 in 4 of 8
* Actuals inside Chronos’s 80 % range: 6 of 8
* Why: Chronos has never seen Mattel. With ~66 quarters of history it should pick up the big Q4 holiday jump, but its middle forecast tends to be smooth, so it will be roughly as good as ETS, not better. B4 is built for seasonal companies like Mattel, so it should stay close. The 80 % range should be roughly honest, but one or two surprise quarters will fall outside.

## Project 3 – Session 2 (all 20 companies)

Saved 19 Nov 2026, before any Chronos-Bolt forecast for the other 19 companies was run.

Claude’s prediction (all 20 companies, 1 quarter ahead, quarters published after 25 Nov 2024)

* MAPE across all 20: Chronos 5.8 %, ETS 5.6 %, B4 6.2 %
* Chronos beats ETS in 9 of 20 companies and beats B4 in 11 of 20
* Actuals inside Chronos’s 80 % range: about 90 % of quarters (too wide, so clearly more than 80 %)
* Chronos too low on average in 13 of 20 companies
* By type: Chronos does best on seasonal and FMCG companies, and worst on growth tech, where it forecasts too low
* Tests: no difference survives the Holm correction
* Why: Chronos has never seen these companies and tends to give smooth, careful forecasts. That suits steady, seasonal revenue but misses fast growth. Mattel’s win came from one lucky quarter, so I expect Chronos and ETS to end up roughly tied across 20 companies.

## Project 3 – Session 3 (base model)

Saved 19 Nov 2026, before any chronos-bolt-base forecast was run.

Claude’s prediction (Session 3 – base model, all 20 companies)

* MAPE across all 20: base 5.0 % (a little better than small’s 5.27 %), still behind ETS 4.29 % and B4 4.23 %
* Base beats ETS for 9 of 20 companies and B4 for 9 of 20
* Deere is still its worst company, at about 15 %
* On seasonal companies it still leans towards “last quarter” (about 0.2, against ETS’s −0.06)
* About 85 % of actuals inside its 10–90 % range
* No test survives Holm, either against ETS and B4 or against the small model
* Why: a bigger model mostly learns the same habits better. The problem in Session 2 was how it handles seasonality, not its size.
