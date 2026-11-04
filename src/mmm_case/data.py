"""Load Meridian's simulated geo dataset into an InputData object.

The CSV is `meridian/data/simulated_data/csv/geo_all_channels.csv` from google/meridian
(Apache-2.0), copied unchanged. It is simulated: no real advertiser's data.
"""

import pandas as pd

from mmm_case.paths import DATA_CSV

PAID_CHANNELS = [f"Channel{i}" for i in range(5)]
CONTROLS = ["sentiment_score_control", "competitor_sales_control"]
NON_MEDIA_TREATMENTS = ["Promo"]
ORGANIC_MEDIA = ["Organic_channel0"]


def load_frame(csv_path=DATA_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=0)


def build_input_data(df: pd.DataFrame | None = None):
    """Same setup as Meridian's Getting Started notebook: conversions KPI with revenue per conversion."""
    from meridian.data import data_frame_input_data_builder

    df = load_frame() if df is None else df
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue",
        default_kpi_column="conversions",
        default_revenue_per_kpi_column="revenue_per_conversion",
    )
    builder = (
        builder.with_kpi(df)
        .with_revenue_per_kpi(df)
        .with_population(df)
        .with_controls(df, control_cols=CONTROLS)
        .with_media(
            df,
            media_cols=[f"{c}_impression" for c in PAID_CHANNELS],
            media_spend_cols=[f"{c}_spend" for c in PAID_CHANNELS],
            media_channels=PAID_CHANNELS,
        )
        .with_non_media_treatments(df, non_media_treatment_cols=NON_MEDIA_TREATMENTS)
        .with_organic_media(
            df,
            organic_media_cols=[f"{c}_impression" for c in ORGANIC_MEDIA],
            organic_media_channels=ORGANIC_MEDIA,
        )
    )
    return builder.build()
