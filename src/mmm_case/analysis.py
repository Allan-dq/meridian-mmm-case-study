"""Turn the fitted model into finance-style numbers.

Every function takes a fitted `Meridian` object and returns plain pandas / dict results, so the
report and the tests read the same values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_case import config
from mmm_case.scenario import reallocate

CI = config.CONFIDENCE_LEVEL
Q_LO, Q_HI = (1 - CI) / 2, 1 - (1 - CI) / 2
METRIC_LABELS = {"mean": "mean", "ci_lo": "ci_low", "ci_hi": "ci_high"}  # Meridian label -> ours


def _analyzer(mmm):
    from meridian.analysis import analyzer

    return analyzer.Analyzer(mmm)


def diagnostics(mmm) -> dict:
    """Convergence (R-hat) and in-sample fit."""
    an = _analyzer(mmm)
    rhat = {k: float(np.nanmax(np.asarray(v))) for k, v in an.get_rhat().items()}
    worst = max(rhat, key=rhat.get)
    acc = an.predictive_accuracy().to_dataframe().reset_index()
    acc = acc[acc["geo_granularity"] == "national"] if "geo_granularity" in acc else acc
    fit = {row["metric"]: float(row["value"]) for _, row in acc.iterrows()}
    return {
        "max_rhat": rhat[worst],
        "max_rhat_parameter": worst,
        "n_parameters_rhat_over_threshold": int(sum(v > config.RHAT_THRESHOLD for v in rhat.values())),
        "rhat_threshold": config.RHAT_THRESHOLD,
        "national_fit": fit,
    }


def review(mmm) -> dict:
    """Meridian's own ModelReviewer health checks."""
    from meridian.analysis.review import reviewer

    summary = reviewer.ModelReviewer(mmm).run()
    checks = []
    for r in summary.results:
        checks.append({
            "check": type(r).__name__.replace("Result", "") if not hasattr(r, "check_name") else r.check_name,
            "status": getattr(getattr(r, "case", None), "status", getattr(r, "status", "")),
            "case": getattr(getattr(r, "case", None), "name", str(getattr(r, "case", ""))),
        })
    return {
        "overall_status": getattr(summary.overall_status, "name", str(summary.overall_status)),
        "summary_message": summary.summary_message,
        "health_score": float(summary.health_score),
        "checks": checks,
    }


def channel_metrics(mmm) -> pd.DataFrame:
    """Spend, incremental revenue, ROI, mROI and cost per incremental KPI, posterior mean and CI."""
    an = _analyzer(mmm)
    ds = an.summary_metrics(confidence_level=CI, include_non_paid_channels=False)
    post = ds.sel(distribution="posterior")
    rows = []
    for ch in post.channel.values:
        p = post.sel(channel=ch)
        row = {"channel": str(ch), "spend": float(p["spend"]), "pct_of_spend": float(p["pct_of_spend"])}
        for var in ["incremental_outcome", "pct_of_contribution", "roi", "mroi", "cpik"]:
            if var in p:
                for m, label in METRIC_LABELS.items():
                    row[f"{var}_{label}"] = float(p[var].sel(metric=m))
        rows.append(row)
    return pd.DataFrame(rows)


def adstock_half_life(mmm) -> pd.DataFrame:
    """Weeks until a week's media effect has decayed to half (posterior mean decay curve)."""
    an = _analyzer(mmm)
    df = an.adstock_decay(confidence_level=CI)
    df = df[df["distribution"] == "posterior"]
    max_lag = int(mmm.model_spec.max_lag)
    rows = []
    for ch, g in df.groupby("channel"):
        g = g.sort_values("time_units")
        w0 = g["mean"].iloc[0]
        below = g[g["mean"] <= 0.5 * w0]
        rows.append({"channel": ch,
                     "half_life_weeks": float(below["time_units"].iloc[0]) if len(below) else np.nan,
                     "max_lag_weeks": max_lag,
                     "weight_after_1_week": float(g.loc[np.isclose(g["time_units"], 1.0), "mean"].iloc[0])})
    return pd.DataFrame(rows)


def response_curves(mmm, multipliers=None) -> pd.DataFrame:
    an = _analyzer(mmm)
    multipliers = multipliers if multipliers is not None else [round(x, 2) for x in np.arange(0, 2.01, 0.1)]
    ds = an.response_curves(spend_multipliers=list(multipliers), confidence_level=CI)
    rows = []
    for ch in ds.channel.values:
        for mult in ds.spend_multiplier.values:
            pt = ds.sel(channel=ch, spend_multiplier=mult)
            row = {"channel": str(ch), "spend_multiplier": float(mult), "spend": float(pt["spend"])}
            for m, label in METRIC_LABELS.items():
                row[label] = float(pt["incremental_outcome"].sel(metric=m))
            rows.append(row)
    return pd.DataFrame(rows)


def _paid_media_and_spend(mmm):
    media = np.asarray(mmm.input_data.media.values, dtype=float)
    spend = np.asarray(mmm.input_data.media_spend.values, dtype=float)
    channels = [str(c) for c in mmm.input_data.media.media_channel.values]
    return media, spend, channels


def _total_incremental_draws(an, new_media=None, new_spend=None) -> np.ndarray:
    """Total paid-media incremental revenue per posterior draw, flattened over chains."""
    from meridian.analysis.tensors import DataTensors

    kwargs = {}
    if new_media is not None:
        kwargs["new_data"] = DataTensors(media=np.asarray(new_media, dtype=np.float32),
                                         media_spend=np.asarray(new_spend, dtype=np.float32))
    draws = np.asarray(an.incremental_outcome(include_non_paid_channels=False, **kwargs))
    return draws.sum(axis=-1).reshape(-1)


def _summarize(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(x, Q_LO)),
        "ci_high": float(np.quantile(x, Q_HI)),
        "prob_positive": float((x > 0).mean()),
    }


def reallocation(mmm, metrics: pd.DataFrame, share: float = config.REALLOCATION_SHARE) -> dict:
    """Move `share` of the lowest-mROI channel's budget to the highest-mROI channel.

    The change in incremental revenue is computed draw by draw (same posterior draw before and
    after), so its credible interval reflects the uncertainty of the difference itself.
    """
    an = _analyzer(mmm)
    media, spend, channels = _paid_media_and_spend(mmm)
    paid = metrics[metrics["channel"].isin(channels)].set_index("channel")
    source = paid["mroi_mean"].idxmin()
    target = paid["mroi_mean"].idxmax()
    base = _total_incremental_draws(an)
    new_media, new_spend = reallocate(media, spend, channels, source, target, share)
    scen = _total_incremental_draws(an, new_media, new_spend)
    diff = scen - base
    moved = share * spend[..., channels.index(source)].sum()
    return {
        "source": source,
        "target": target,
        "share": share,
        "spend_moved": float(moved),
        "total_spend": float(spend.sum()),
        "base_incremental_revenue": _summarize(base),
        "change_in_incremental_revenue": _summarize(diff),
        "change_per_dollar_moved": float(diff.mean() / moved),
        "source_mroi": float(paid.loc[source, "mroi_mean"]),
        "target_mroi": float(paid.loc[target, "mroi_mean"]),
    }


def reallocation_matrix(mmm, share: float = config.REALLOCATION_SHARE) -> pd.DataFrame:
    """Mean change in incremental revenue for every ordered (source, target) pair."""
    an = _analyzer(mmm)
    media, spend, channels = _paid_media_and_spend(mmm)
    base = _total_incremental_draws(an)
    rows = []
    for s in channels:
        for t in channels:
            if s == t:
                continue
            nm, ns = reallocate(media, spend, channels, s, t, share)
            d = _total_incremental_draws(an, nm, ns) - base
            rows.append({"source": s, "target": t, **_summarize(d)})
    return pd.DataFrame(rows)


def optimize_budget(mmm) -> dict:
    """Meridian's fixed-budget optimizer with its default per-channel bounds (spend within +/-30%)."""
    from meridian.analysis import optimizer

    res = optimizer.BudgetOptimizer(mmm).optimize(fixed_budget=True, confidence_level=CI)
    non, opt = res.nonoptimized_data, res.optimized_data
    rows = []
    for ch in opt.channel.values:
        rows.append({
            "channel": str(ch),
            "current_spend": float(non["spend"].sel(channel=ch)),
            "optimized_spend": float(opt["spend"].sel(channel=ch)),
            "current_incremental_revenue": float(non["incremental_outcome"].sel(channel=ch, metric="mean")),
            "optimized_incremental_revenue": float(opt["incremental_outcome"].sel(channel=ch, metric="mean")),
        })
    table = pd.DataFrame(rows)
    table["spend_change_pct"] = table["optimized_spend"] / table["current_spend"] - 1

    # Lift draw by draw: apply each channel's optimized spend multiplier to its historical media.
    an = _analyzer(mmm)
    media, spend, channels = _paid_media_and_spend(mmm)
    factors = (table.set_index("channel").loc[channels, "optimized_spend"]
               / table.set_index("channel").loc[channels, "current_spend"]).to_numpy()
    base = _total_incremental_draws(an)
    lift = _total_incremental_draws(an, media * factors, spend * factors) - base
    return {
        "table": table,
        "total_budget": float(table["current_spend"].sum()),
        "optimizer_reported_lift": float(table["optimized_incremental_revenue"].sum()
                                         - table["current_incremental_revenue"].sum()),
        "lift_draws": _summarize(lift),
        "spend_bounds": "each channel within +/-30% of its current spend (Meridian default)",
    }


def budget_level(mmm, factor: float = 0.9) -> dict:
    """Scale every channel's spend (and impressions) by `factor`: what does the last slice of budget buy?"""
    an = _analyzer(mmm)
    media, spend, _ = _paid_media_and_spend(mmm)
    base = _total_incremental_draws(an)
    diff = _total_incremental_draws(an, media * factor, spend * factor) - base
    spend_change = float(spend.sum() * (factor - 1))
    return {
        "factor": factor,
        "spend_change": spend_change,
        "change_in_incremental_revenue": _summarize(diff),
        "revenue_per_dollar_of_spend_change": float(diff.mean() / spend_change),
        "prob_revenue_change_smaller_than_spend_change": float((diff > spend_change).mean()),
    }
