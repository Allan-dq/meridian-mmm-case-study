"""Finance numbers from the fitted PyMC-Marketing model, defined exactly as in analysis.py.

Every scenario is evaluated draw by draw on the same posterior draws, so a difference between two
scenarios carries the uncertainty of the difference itself. The model has an identity link, so a
channel's effect does not depend on the other channels: the effect of changing several channels
at once is the sum of the one-channel effects, which is how the budget-level and reallocation
scenarios below are built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_case import config
from mmm_case.analysis import _summarize
from mmm_case.data import PAID_CHANNELS
from mmm_case.pymc_model import GEO, TARGET

CI = config.CONFIDENCE_LEVEL
MARGINAL_INCREASE = 0.01      # Meridian's mROI uses a 1% spend increase; so does this
DECAY_GRID = np.round(np.arange(0, 8.0 + 1e-9, 0.1), 1)


def revenue_change(mmm, factor: float) -> pd.DataFrame:
    """National change in revenue, R(spend x factor) - R(actual), for each channel on its own.

    Returns a (draws, channels) frame. PyMC-Marketing reports both directions as positive numbers
    (actual minus counterfactual for factor <= 1, the reverse above 1); this function undoes that
    so that a cut is negative and an increase positive.
    """
    da = mmm.incrementality.compute_incremental_contribution(
        frequency="all_time", counterfactual_spend_factor=factor
    )
    da = da.sum(GEO).stack(sample=("chain", "draw")).transpose("sample", "channel")
    df = pd.DataFrame(np.asarray(da), columns=[str(c) for c in da["channel"].values])
    return df if factor > 1 else -df


def _ci(x: np.ndarray, prefix: str) -> dict:
    s = _summarize(np.asarray(x))
    return {f"{prefix}_mean": s["mean"], f"{prefix}_ci_low": s["ci_low"], f"{prefix}_ci_high": s["ci_high"]}


def channel_metrics(mmm, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spend, incremental revenue, contribution share, ROI and mROI per paid channel, plus a total row.

    Also returns the per-draw ROI frame (for charts and tests).
    """
    spend = frame[PAID_CHANNELS].sum()
    total_revenue = frame[TARGET].sum()
    incremental = -revenue_change(mmm, 0.0)[PAID_CHANNELS]
    marginal = revenue_change(mmm, 1 + MARGINAL_INCREASE)[PAID_CHANNELS]
    rows = []
    for ch in PAID_CHANNELS:
        inc = incremental[ch].to_numpy()
        rows.append({
            "channel": ch,
            "spend": float(spend[ch]),
            "pct_of_spend": float(100 * spend[ch] / spend.sum()),
            **_ci(inc, "incremental_outcome"),
            **_ci(100 * inc / total_revenue, "pct_of_contribution"),
            **_ci(inc / spend[ch], "roi"),
            **_ci(marginal[ch].to_numpy() / (MARGINAL_INCREASE * spend[ch]), "mroi"),
        })
    total = incremental.sum(axis=1).to_numpy()
    rows.append({
        "channel": "All Channels",
        "spend": float(spend.sum()),
        "pct_of_spend": 100.0,
        **_ci(total, "incremental_outcome"),
        **_ci(100 * total / total_revenue, "pct_of_contribution"),
        **_ci(total / spend.sum(), "roi"),
    })
    roi_draws = incremental / spend
    return pd.DataFrame(rows), roi_draws


def budget_level(mmm, frame: pd.DataFrame, factor: float = 0.9) -> dict:
    """Scale every paid channel's spend by `factor` (the organic channel is left as it is)."""
    spend = frame[PAID_CHANNELS].sum()
    diff = revenue_change(mmm, factor)[PAID_CHANNELS].sum(axis=1).to_numpy()
    spend_change = float(spend.sum() * (factor - 1))
    return {
        "factor": factor,
        "spend_change": spend_change,
        "change_in_incremental_revenue": _summarize(diff),
        "revenue_per_dollar_of_spend_change": float(diff.mean() / spend_change),
        "prob_revenue_change_smaller_than_spend_change": float((diff > spend_change).mean()),
    }


def reallocation(mmm, frame: pd.DataFrame, source: str, target: str,
                 share: float = config.REALLOCATION_SHARE) -> dict:
    """Move `share` of `source`'s budget to `target` (same rule as scenario.reallocate)."""
    spend = frame[PAID_CHANNELS].sum()
    moved = share * spend[source]
    target_factor = 1 + moved / spend[target]
    diff = (revenue_change(mmm, 1 - share)[source] + revenue_change(mmm, target_factor)[target]).to_numpy()
    return {
        "source": source,
        "target": target,
        "share": share,
        "spend_moved": float(moved),
        "change_in_incremental_revenue": _summarize(diff),
        "change_per_dollar_moved": float(diff.mean() / moved),
    }


def adstock_half_life(mmm) -> pd.DataFrame:
    """Weeks until the posterior-mean decay curve falls to half its lag-0 weight (as in analysis.py)."""
    alpha = mmm.idata.posterior["adstock_alpha"].stack(sample=("chain", "draw")).transpose("sample", "channel")
    a = np.asarray(alpha)
    curve = (a[:, :, None] ** DECAY_GRID[None, None, :]).mean(axis=0)      # (channel, lag)
    rows = []
    for i, ch in enumerate(alpha["channel"].values):
        below = np.nonzero(curve[i] <= 0.5 * curve[i, 0])[0]
        rows.append({
            "channel": str(ch),
            "half_life_weeks": float(DECAY_GRID[below[0]]) if below.size else np.nan,
            "weight_after_1_week": float(a[:, i].mean()),
        })
    return pd.DataFrame(rows)
