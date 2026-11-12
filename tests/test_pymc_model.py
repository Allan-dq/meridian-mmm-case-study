"""Checks on the PyMC-Marketing comparison helpers. They need NumPy, pandas and SciPy only, so they
run in the Meridian environment and in CI; fitting the model itself needs requirements-pymc.txt."""

import numpy as np
import pytest

from mmm_case import config
from mmm_case import pymc_model as pmm
from mmm_case.data import PAID_CHANNELS, load_frame


@pytest.fixture(scope="module")
def frame():
    return pmm.model_frame()


def test_adstock_impulse_response_is_the_normalized_geometric_weights():
    x = np.zeros((1, 20))
    x[0, 0] = 1.0
    out = pmm.geometric_adstock(x, np.array([0.5]), l_max=9)[0, 0]
    w = 0.5 ** np.arange(9)
    np.testing.assert_allclose(out[:9], w / w.sum())
    assert np.all(out[9:] == 0)


def test_adstock_keeps_a_constant_series_constant_once_the_window_is_full():
    x = np.full((2, 30), 3.0)
    out = pmm.geometric_adstock(x, np.array([0.2, 0.9]))
    np.testing.assert_allclose(out[:, :, pmm.L_MAX - 1:], 3.0)


def test_model_frame_keeps_every_row_and_total(frame):
    raw = load_frame()
    assert len(frame) == len(raw) == 40 * 156
    assert frame[pmm.TARGET].sum() == pytest.approx((raw["conversions"] * raw["revenue_per_conversion"]).sum())
    for ch in PAID_CHANNELS:
        assert frame[ch].sum() == pytest.approx(raw[f"{ch}_spend"].sum())


def test_spend_and_impressions_differ_by_a_constant_factor_per_channel():
    # The reason spend can stand in for impressions in the PyMC-Marketing model.
    raw = load_frame()
    for ch in PAID_CHANNELS:
        on = raw[f"{ch}_impression"] > 0
        ratio = raw.loc[on, f"{ch}_spend"] / raw.loc[on, f"{ch}_impression"]
        assert ratio.std() / ratio.mean() < 1e-4
        assert (raw.loc[~on, f"{ch}_spend"] == 0).all()


def test_more_spend_for_the_same_pattern_lowers_log_roi_one_for_one(frame):
    target_max = pmm.panel(frame, pmm.TARGET).max(axis=1)
    spend = pmm.panel(frame, "Channel0")
    a = pmm.implied_log_roi(spend, target_max, 0.5, n=2000, seed=1)
    b = pmm.implied_log_roi(2 * spend, target_max, 0.5, n=2000, seed=1)
    np.testing.assert_allclose(b, a - np.log(2))


def test_calibrated_prior_matches_meridians_roi_prior(frame):
    cal = pmm.calibrate_roi_prior(frame, n=4000)
    assert list(cal["channel"]) == PAID_CHANNELS
    np.testing.assert_allclose(cal["prior_roi_median"], np.exp(config.ROI_PRIOR_MU), rtol=1e-9)
    np.testing.assert_allclose(cal["prior_log_roi_sd"], config.ROI_PRIOR_SIGMA, atol=0.05)
    assert (cal["prior_roi_p05"] < cal["prior_roi_median"]).all()
    assert (cal["prior_roi_median"] < cal["prior_roi_p95"]).all()


def test_half_saturation_prior_is_meridians_rescaled():
    p = pmm.half_saturation_prior_params(0.25)
    assert p == pytest.approx({"mu": 0.2, "sigma": 0.2, "lower": 0.025, "upper": 2.5})
