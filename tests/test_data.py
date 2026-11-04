import numpy as np

from mmm_case import data


def test_csv_has_expected_columns_and_no_negative_spend():
    df = data.load_frame()
    for c in data.PAID_CHANNELS:
        assert f"{c}_impression" in df and f"{c}_spend" in df
        assert (df[f"{c}_spend"] >= 0).all()
        assert (df[f"{c}_impression"] >= 0).all()
    assert df[["geo", "time"]].duplicated().sum() == 0


def test_panel_is_complete():
    df = data.load_frame()
    n_geo, n_time = df["geo"].nunique(), df["time"].nunique()
    assert len(df) == n_geo * n_time


def test_input_data_builds_with_all_paid_channels():
    idata = data.build_input_data()
    assert [str(c) for c in idata.media.media_channel.values] == data.PAID_CHANNELS
    assert np.isfinite(np.asarray(idata.media_spend.values)).all()
