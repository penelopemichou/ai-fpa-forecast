"""Part 2, Session 1: a pretrained forecasting model (Chronos-Bolt small) on Mattel.

What a pretrained model is: Amazon trained Chronos-Bolt on about 100 billion
time-series observations from many sources. We do NOT train it on Mattel. We
just hand it Mattel's known quarters (the "context") and it returns a forecast
straight away ("zero-shot").

Fixed before any Mattel forecast was made (19 Nov 2026):
  - Model: amazon/chronos-bolt-small (48M parameters), CPU, float32,
    pinned to Hugging Face revision REVISION below.
  - Release date: the model card states none. The Hugging Face repo's first
    commit is 13 Nov 2024 and the repo was created on 25 Nov 2024. We use the
    later date, 25 Nov 2024.
  - Quarters scored: one quarter ahead (h = 1), target quarters whose actual
    revenue was first PUBLISHED (first_filed_date) after the release date:
    2024 Q4 to 2026 Q3 (8 quarters). Changed on 19 Nov 2026, before the
    first run, from "quarters that start after the release" (7 quarters).
  - Context: exactly the quarters ETS used at each origin (first-reported
    revenue, 65-day rule, `training_series` from src/ets.py).
  - Forecast: the median (0.5 quantile) of Chronos-Bolt's forecast.
  - Compared with ETS (results/ets_forecasts.csv) and B4
    (results/session7_b4_forecasts.csv) on exactly the same 7 targets.
  - Release date is not the training-data cutoff (the card names neither its
    datasets nor a cutoff). Scoring only quarters after the release means the
    model cannot have seen the quarters it is scored on.

Run from the project folder:
    .venv/bin/python src/chronos_bolt.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from baselines import cutoff_date, known_at, load
from ets import training_series

MODEL = "amazon/chronos-bolt-small"
REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"   # latest on 19 Nov 2026
RELEASE_DATE = pd.Timestamp("2024-11-25")
COMPANY = "Mattel"
H = 1
OUT = Path("results/chronos_mattel_forecasts.csv")


def scored_targets(cdf):
    """Mattel's fiscal quarters whose actual was first published after the release date."""
    filed = cdf.set_index("quarter")["first_filed_date"]
    return [q for q, d in filed.items() if d > RELEASE_DATE]


def chronos_forecasts(pipeline, cdf):
    rows = []
    for target in scored_targets(cdf):
        origin = target - H
        known = known_at(cdf, origin)
        context = training_series(known)               # same quarters ETS used
        assert context.index[-1] == origin, "latest quarter not known at origin"
        assert (cdf.set_index("quarter").loc[context.index, "first_filed_date"]
                <= cutoff_date(origin, cdf)).all(), "leak: a context quarter was not public yet"
        quantiles, _ = pipeline.predict_quantiles(
            torch.tensor(context.to_numpy() / 1e9, dtype=torch.float32),
            prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
        q = quantiles[0, H - 1].numpy() * 1e9
        rows.append({"company": COMPANY, "origin": str(origin), "h": H, "target": str(target),
                     "actual": cdf.set_index("quarter").loc[target, "value"],
                     "actual_filed": cdf.set_index("quarter").loc[target, "first_filed_date"].date(),
                     "chronos": q[1], "chronos_q10": q[0], "chronos_q90": q[2],
                     "n_context": len(context), "context_start": str(context.index[0])})
    return pd.DataFrame(rows)


def compare(chronos):
    ets = pd.read_csv("results/ets_forecasts.csv")
    ets = ets[(ets.company == COMPANY) & (ets.h == H) & (ets.series == "first_reported")]
    b4 = pd.read_csv("results/session7_b4_forecasts.csv")
    b4 = b4[(b4.company == COMPANY) & (b4.h == H)]
    df = (chronos.merge(ets[["target", "ets", "n_train"]], on="target")
                 .merge(b4[["target", "b4"]], on="target"))
    assert len(df) == len(chronos), "ETS or B4 is missing a target quarter"
    assert (df.n_train == df.n_context).all(), "Chronos and ETS saw different quarters"
    for m in ["chronos", "ets", "b4"]:
        df[f"ape_{m}"] = 100 * (df[m] - df.actual).abs() / df.actual
    return df


def main():
    torch.manual_seed(0)
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL, revision=REVISION, device_map="cpu", torch_dtype=torch.float32)
    cdf = load()
    cdf = cdf[cdf.company == COMPANY].reset_index(drop=True)
    df = compare(chronos_forecasts(pipeline, cdf))
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 140)
    show = df[["target", "actual_filed", "actual", "chronos", "chronos_q10", "chronos_q90",
               "ets", "b4", "ape_chronos", "ape_ets", "ape_b4"]].copy()
    for c in ["actual", "chronos", "chronos_q10", "chronos_q90", "ets", "b4"]:
        show[c] = (show[c] / 1e6).round(0).astype(int)     # $ millions
    print(f"Mattel, one quarter ahead ($ millions; ape = % miss). Release date {RELEASE_DATE.date()}:")
    print(show.round(1).to_string(index=False))
    print("\nMAPE over the same", len(df), "quarters:")
    for m in ["chronos", "ets", "b4"]:
        print(f"  {m:8s} {df[f'ape_{m}'].mean():5.2f} %")
    print("\nChronos closer than ETS in", int((df.ape_chronos < df.ape_ets).sum()), "of", len(df),
          "| closer than B4 in", int((df.ape_chronos < df.ape_b4).sum()), "of", len(df))
    inside = ((df.actual >= df.chronos_q10) & (df.actual <= df.chronos_q90)).sum()
    print(f"Actual inside Chronos's 10-90 % range: {inside} of {len(df)}")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
