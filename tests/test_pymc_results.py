"""Consistency checks on the committed PyMC-Marketing comparison (skipped until compare_pymc.py has run)."""

import json

import numpy as np
import pandas as pd
import pytest

from mmm_case import config, paths
from mmm_case import pymc_model as pmm
from mmm_case.data import PAID_CHANNELS

PYMC = paths.REPORTS / "pymc"
RESULTS = PYMC / "results.json"
pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="run scripts/compare_pymc.py first")


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(PYMC / "channel_metrics.csv")


def test_intervals_contain_the_mean(metrics):
    for var in ["roi", "incremental_outcome"]:
        assert (metrics[f"{var}_ci_low"] <= metrics[f"{var}_mean"]).all()
        assert (metrics[f"{var}_mean"] <= metrics[f"{var}_ci_high"]).all()
    paid = metrics[metrics["channel"].isin(PAID_CHANNELS)]
    assert ((paid["mroi_ci_low"] <= paid["mroi_mean"]) & (paid["mroi_mean"] <= paid["mroi_ci_high"])).all()


def test_spend_is_the_same_as_in_the_meridian_report(metrics):
    mer = pd.read_csv(paths.REPORTS / "channel_metrics.csv").set_index("channel")
    ours = metrics.set_index("channel")
    for ch in [*PAID_CHANNELS, "All Channels"]:
        assert ours.loc[ch, "spend"] == pytest.approx(mer.loc[ch, "spend"], rel=1e-6)


def test_roi_is_incremental_revenue_over_spend(metrics):
    np.testing.assert_allclose(metrics["roi_mean"], metrics["incremental_outcome_mean"] / metrics["spend"], rtol=1e-9)


def test_budget_cut_and_reallocation_move_the_configured_amounts(results, metrics):
    total = metrics.set_index("channel").loc["All Channels", "spend"]
    assert results["budget_level"]["spend_change"] == pytest.approx(-0.1 * total, rel=1e-9)
    r = results["reallocation_same_as_meridian"]
    src = metrics.set_index("channel").loc[r["source"], "spend"]
    assert r["spend_moved"] == pytest.approx(config.REALLOCATION_SHARE * src, rel=1e-9)


def test_sampler_converged(results):
    fit = results["fit"]
    n_draws = fit["sampler"]["chains"] * fit["sampler"]["draws"]
    assert fit["max_rhat"] < 1.1
    assert fit["divergences"] < 0.01 * n_draws
    assert fit["min_ess_bulk"] > 50


def test_knot_weights_match_meridian():
    knots = pmm.load_meridian_knots()
    w = pmm.knot_weights(156, knots)
    np.testing.assert_allclose(w.sum(axis=0), 1.0)
    try:
        from meridian.model import knots as mknots
    except ImportError:
        pytest.skip("Meridian not installed in this environment")
    np.testing.assert_allclose(w, mknots.l1_distance_weights(156, np.asarray(knots)), atol=1e-6)
