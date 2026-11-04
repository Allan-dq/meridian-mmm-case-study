"""Check google/meridian#1751: does a serde save/load round trip keep the paid-media prior type?

Fits tiny models (a few MCMC draws: we test serialization, not model quality) with different
prior-type settings, saves each with `meridian_serde.save_meridian`, loads it back, and compares
the ModelSpec prior-type fields and the posterior variable names before and after.

Usage: python scripts/check_issue_1751.py data/geo_all_channels.csv
"""

import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

import meridian
from meridian.data import data_frame_input_data_builder
from meridian.model import model, spec
from meridian.schema.serde import meridian_serde

FIELDS = [
    "media_prior_type",
    "rf_prior_type",
    "paid_media_prior_type",
    "effective_media_prior_type",
    "effective_rf_prior_type",
]
CONFIGS = {
    "deprecated paid_media_prior_type='contribution'": {"paid_media_prior_type": "contribution"},
    "media_prior_type='contribution'": {"media_prior_type": "contribution"},
    "media_prior_type='mroi'": {"media_prior_type": "mroi"},
    "default (roi)": {},
}


def build_data(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    channels = [f"Channel{i}" for i in range(5)]
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue",
        default_kpi_column="conversions",
        default_revenue_per_kpi_column="revenue_per_conversion",
    )
    builder = (
        builder.with_kpi(df)
        .with_revenue_per_kpi(df)
        .with_population(df)
        .with_controls(df, control_cols=["sentiment_score_control", "competitor_sales_control"])
        .with_media(
            df,
            media_cols=[f"{c}_impression" for c in channels],
            media_spend_cols=[f"{c}_spend" for c in channels],
            media_channels=channels,
        )
    )
    return builder.build()


def spec_fields(model_spec) -> dict:
    return {f: getattr(model_spec, f, "<missing>") for f in FIELDS}


def media_vars(mmm) -> list[str]:
    names = set(mmm.inference_data.posterior.data_vars)
    return sorted(n for n in names if n in {"roi_m", "mroi_m", "contribution_m", "beta_m"})


def main(csv_path: str) -> int:
    print(f"meridian {meridian.__version__}")
    data = build_data(csv_path)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, kwargs in CONFIGS.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model_spec = spec.ModelSpec(**kwargs)
                mmm = model.Meridian(input_data=data, model_spec=model_spec)
                mmm.sample_prior(20, seed=1)
                mmm.sample_posterior(n_chains=1, n_adapt=20, n_burnin=20, n_keep=20, seed=1)
                path = str(Path(tmp) / "model.binpb")
                meridian_serde.save_meridian(mmm, path)
                loaded = meridian_serde.load_meridian(path)
            before, after = spec_fields(mmm.model_spec), spec_fields(loaded.model_spec)
            vars_before, vars_after = media_vars(mmm), media_vars(loaded)
            ok = before == after and vars_before == vars_after
            failures += not ok
            print(f"\n[{'OK  ' if ok else 'FAIL'}] {name}")
            for f in FIELDS:
                mark = "" if before[f] == after[f] else "   <-- changed"
                print(f"    {f:28s} before={before[f]!s:14s} after={after[f]!s:14s}{mark}")
            print(f"    posterior media vars        before={vars_before} after={vars_after}")
            prior_warnings = sorted({str(w.message)[:90] for w in caught if "prior type" in str(w.message).lower()})
            for msg in prior_warnings:
                print(f"    warning: {msg}")
    print(f"\n{failures} of {len(CONFIGS)} configurations changed after the round trip.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
