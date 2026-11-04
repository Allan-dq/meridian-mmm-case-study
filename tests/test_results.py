"""Consistency checks on the committed results (skipped until scripts/analyze.py has run)."""

import json

import pandas as pd
import pytest

from mmm_case import config, paths

RESULTS = paths.REPORTS / "results.json"
pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="run `make all` first")


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(paths.REPORTS / "channel_metrics.csv")


def test_intervals_contain_the_mean(metrics):
    for var in ["roi", "incremental_outcome"]:
        assert (metrics[f"{var}_ci_low"] <= metrics[f"{var}_mean"]).all()
        assert (metrics[f"{var}_mean"] <= metrics[f"{var}_ci_high"]).all()
    paid = metrics[~metrics["channel"].str.startswith("All")]
    assert (paid["mroi_ci_low"] <= paid["mroi_mean"]).all() and (paid["mroi_mean"] <= paid["mroi_ci_high"]).all()


def test_channel_spend_adds_up_to_total(metrics):
    paid = metrics[~metrics["channel"].str.startswith("All")]
    total = metrics[metrics["channel"].str.startswith("All")]["spend"].iloc[0]
    assert paid["spend"].sum() == pytest.approx(total, rel=1e-6)


def test_reallocation_moves_the_configured_share(results, metrics):
    r = results["reallocation"]
    src_spend = metrics.set_index("channel").loc[r["source"], "spend"]
    assert r["spend_moved"] == pytest.approx(config.REALLOCATION_SHARE * src_spend, rel=1e-6)
    assert r["source_mroi"] < r["target_mroi"]


def test_optimizer_keeps_the_budget_and_our_lift_matches_its_estimate(results):
    table = pd.read_csv(paths.REPORTS / "budget_optimization.csv")
    assert table["optimized_spend"].sum() == pytest.approx(table["current_spend"].sum(), rel=1e-3)
    assert (table["spend_change_pct"].abs() <= 0.30 + 1e-6).all()
    o = results["optimization"]
    assert o["lift_draws"]["mean"] == pytest.approx(o["optimizer_reported_lift"], rel=0.02)


def test_model_converged(results):
    d = results["diagnostics"]
    assert d["max_rhat"] < d["rhat_threshold"], f"max R-hat {d['max_rhat']:.3f} on {d['max_rhat_parameter']}"
