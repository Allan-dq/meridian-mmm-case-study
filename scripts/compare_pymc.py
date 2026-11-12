"""Compare the PyMC-Marketing fit with the Meridian results already in reports/.

    .venv-pymc/bin/python scripts/compare_pymc.py
    .venv-pymc/bin/python scripts/compare_pymc.py --model smoke/pymc_model.nc --out-dir smoke/pymc

Reads artifacts/pymc_model.nc and reports/pymc/{fit_info.json,prior_calibration.csv} (written by
fit_pymc.py) and Meridian's reports/{channel_metrics.csv,adstock_half_life.csv,results.json,
fit_info.json}. Writes reports/pymc/{channel_metrics.csv,adstock_half_life.csv,results.json,
comparison.md} and reports/figures/meridian_vs_pymc_roi.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mmm_case import charts, config, paths  # noqa: E402
from mmm_case import pymc_analysis as pa  # noqa: E402
from mmm_case import pymc_model as pmm  # noqa: E402
from mmm_case.data import PAID_CHANNELS  # noqa: E402


def money(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x) / 1e6:,.1f}M"


def ci(row, var, fmt="{:.2f}") -> str:
    return f"{fmt.format(row[f'{var}_mean'])} [{fmt.format(row[f'{var}_ci_low'])}, {fmt.format(row[f'{var}_ci_high'])}]"


def ci_money(s: dict) -> str:
    return f"{money(s['mean'])} [{money(s['ci_low'])}, {money(s['ci_high'])}]"


def ranking(m: pd.DataFrame, col: str) -> list[str]:
    paid = m[m["channel"].isin(PAID_CHANNELS)]
    return paid.sort_values(f"{col}_mean", ascending=False)["channel"].tolist()


def spearman(a: list[str], b: list[str]) -> float:
    ra = {c: i for i, c in enumerate(a)}
    rb = {c: i for i, c in enumerate(b)}
    d2 = sum((ra[c] - rb[c]) ** 2 for c in a)
    n = len(a)
    return 1 - 6 * d2 / (n * (n**2 - 1))


SETUP = """| | Meridian 2.1.0 | PyMC-Marketing 1.4.0 |
|---|---|---|
| Data | 40 geos x 156 weeks, revenue = conversions x revenue per conversion | same rows and columns |
| Paid media input | impressions (spend used for ROI) | spend; cost per impression is constant per channel here, so the two differ by a fixed factor |
| Carryover | geometric adstock, lags 0-8, decay ~ Uniform(0, 1) | same (`GeometricAdstock(l_max=9)`, decay ~ Beta(1, 1)) |
| Saturation | Hill, slope fixed at 1, half-saturation ~ TruncatedNormal(0.8, 0.8, 0.1, 10) x median media | same curve (Michaelis-Menten); the same prior, rescaled to the max-scaled inputs |
| Geo pooling | channel effect log-normal across geos around a national level, spread ~ HalfNormal(1) | same |
| Scaling | per capita | each geo divided by its own maximum (`DataDerivedScaling(dims=())`); the library default for a geo model is one national maximum |
| Time effect | national, 66 knots chosen automatically, linear interpolation, knot values ~ Normal(0, 5) | the same 66 knots and weights as a custom additive effect, each basis column centred over the fit period, knot values ~ zero-sum Normal(0, 0.5) on the max-scaled target |
| Controls | 2 controls, the promotion as a non-media treatment, the organic channel with its own adstock and Hill curve | 2 controls and the promotion as controls (geo-varying around a national mean), the organic channel as a media channel without ROI |
| Noise | one Normal scale for all geos | same |
| ROI prior | ROI ~ LogNormal(0.2, 0.9) on every paid channel (Meridian's ROI parameterization) | no ROI parameterization: each channel's effect prior is set so that the ROI it implies has the same median (1.22) and log-scale sd (0.90); see `prior_calibration.csv` |
| Sampler | NUTS (TensorFlow Probability on JAX), 4 chains x 1,000 kept draws | NUTS (nutpie), 4 chains x 1,000 draws after 1,500 tuning steps, target acceptance 0.99 |"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.ARTIFACTS / "pymc_model.nc"))
    ap.add_argument("--out-dir", default=str(paths.REPORTS / "pymc"))
    ap.add_argument("--figure", default=str(paths.FIGURES / "meridian_vs_pymc_roi.png"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import mmm_case.pymc_effects  # noqa: F401  (registers KnotTimeEffect for MMM.load)
    from pymc_marketing.mmm import MMM

    t0 = time.time()
    mmm = MMM.load(args.model)
    frame = pmm.model_frame()

    metrics, roi_draws = pa.channel_metrics(mmm, frame)
    half_life = pa.adstock_half_life(mmm)
    level = pa.budget_level(mmm, frame, 0.9)

    mer_metrics = pd.read_csv(paths.REPORTS / "channel_metrics.csv")
    mer_half = pd.read_csv(paths.REPORTS / "adstock_half_life.csv")
    mer = json.loads((paths.REPORTS / "results.json").read_text())
    mer_fit = json.loads((paths.REPORTS / "fit_info.json").read_text())
    fit = json.loads((out / "fit_info.json").read_text())
    calibration = pd.read_csv(out / "prior_calibration.csv")

    # Same move as the Meridian report, then PyMC-Marketing's own lowest -> highest mROI pair.
    m_src, m_tgt = mer["reallocation"]["source"], mer["reallocation"]["target"]
    realloc_same = pa.reallocation(mmm, frame, m_src, m_tgt)
    paid = metrics[metrics["channel"].isin(PAID_CHANNELS)].set_index("channel")
    p_src, p_tgt = paid["mroi_mean"].idxmin(), paid["mroi_mean"].idxmax()
    realloc_own = realloc_same if (p_src, p_tgt) == (m_src, m_tgt) else pa.reallocation(mmm, frame, p_src, p_tgt)

    metrics.to_csv(out / "channel_metrics.csv", index=False)
    half_life.to_csv(out / "adstock_half_life.csv", index=False)
    charts.model_comparison(mer_metrics, metrics, args.figure, config.CONFIDENCE_LEVEL)

    rank_roi = (ranking(mer_metrics, "roi"), ranking(metrics, "roi"))
    rank_mroi = (ranking(mer_metrics, "mroi"), ranking(metrics, "mroi"))
    mer_paid = mer_metrics[mer_metrics["channel"].isin(PAID_CHANNELS)]
    agreement = {
        "roi_rank_meridian": rank_roi[0],
        "roi_rank_pymc": rank_roi[1],
        "roi_rank_spearman": spearman(*rank_roi),
        "mroi_rank_meridian": rank_mroi[0],
        "mroi_rank_pymc": rank_mroi[1],
        "mroi_rank_spearman": spearman(*rank_mroi),
        "all_mroi_below_1_meridian": bool((mer_paid["mroi_mean"] < 1).all()),
        "all_mroi_below_1_pymc": bool((paid["mroi_mean"] < 1).all()),
        "max_abs_roi_difference": float(
            (mer_paid.set_index("channel")["roi_mean"] - paid["roi_mean"]).abs().max()),
        # How often one model's posterior mean falls inside the other's interval, per channel.
        "pymc_roi_mean_inside_meridian_interval": int(sum(
            mer_paid.set_index("channel").loc[c, "roi_ci_low"] <= paid.loc[c, "roi_mean"]
            <= mer_paid.set_index("channel").loc[c, "roi_ci_high"] for c in PAID_CHANNELS)),
    }
    results = {
        "fit": fit,
        "channel_metrics": metrics.to_dict("records"),
        "budget_level": level,
        "reallocation_same_as_meridian": realloc_same,
        "reallocation_own_pick": realloc_own,
        "agreement": agreement,
        "confidence_level": config.CONFIDENCE_LEVEL,
        "analysis_runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # ---------- comparison.md ----------
    pct = int(config.CONFIDENCE_LEVEL * 100)
    mm, pm_ = mer_metrics.set_index("channel"), metrics.set_index("channel")
    mh, ph = mer_half.set_index("channel"), half_life.set_index("channel")
    lines = [
        "# Meridian vs PyMC-Marketing on the same data (generated by scripts/compare_pymc.py; do not edit by hand)",
        "",
        f"Both models are Bayesian marketing-mix models fitted to Meridian's simulated `geo_all_channels.csv`. "
        f"Intervals are {pct}% posterior intervals. The data are simulated, and the true channel effects are not "
        "published, so this compares two models with each other, not with the truth.",
        "",
        "## How the two models are set up",
        "",
        SETUP,
        "",
        "## Convergence and fit",
        "",
        "| | Meridian | PyMC-Marketing |",
        "|---|---:|---:|",
        f"| Max R-hat (all parameters) | {mer['diagnostics']['max_rhat']:.3f} | {fit['max_rhat']:.3f} |",
        f"| Divergent transitions | not reported by Meridian | {fit['divergences']} |",
        f"| Smallest bulk effective sample size | not computed | {fit['min_ess_bulk']:.0f} |",
        f"| Sampling time on this laptop | {mer_fit['runtime_seconds'] / 60:.0f} min | "
        f"{fit['runtime_seconds'] / 60:.1f} min |",
        f"| National weekly revenue: R-squared | {mer['diagnostics']['national_fit']['R_Squared']:.3f} | "
        f"{fit['national_fit']['R_Squared']:.3f} |",
        f"| National weekly revenue: MAPE | {mer['diagnostics']['national_fit']['MAPE']:.1%} | "
        f"{fit['national_fit']['MAPE']:.1%} |",
        "",
        "## ROI and mROI by channel",
        "",
        "| Channel | Share of spend | ROI: Meridian | ROI: PyMC-Marketing | mROI: Meridian | mROI: PyMC-Marketing |",
        "|---|---:|---|---|---|---|",
    ]
    for ch in PAID_CHANNELS:
        lines.append(f"| {ch} | {mm.loc[ch, 'pct_of_spend']:.0f}% | {ci(mm.loc[ch], 'roi')} | {ci(pm_.loc[ch], 'roi')} | "
                     f"{ci(mm.loc[ch], 'mroi')} | {ci(pm_.loc[ch], 'mroi')} |")
    allm, allp = mm.loc["All Channels"], pm_.loc["All Channels"]
    lines += [
        f"| **All paid channels** | 100% | {ci(allm, 'roi')} | {ci(allp, 'roi')} | | |",
        "",
        f"Incremental revenue from paid media: Meridian {money(allm['incremental_outcome_mean'])} "
        f"[{money(allm['incremental_outcome_ci_low'])}, {money(allm['incremental_outcome_ci_high'])}], "
        f"PyMC-Marketing {money(allp['incremental_outcome_mean'])} "
        f"[{money(allp['incremental_outcome_ci_low'])}, {money(allp['incremental_outcome_ci_high'])}], "
        f"on {money(allm['spend'])} of spend "
        f"({allm['pct_of_contribution_mean']:.1f}% and {allp['pct_of_contribution_mean']:.1f}% of revenue).",
        "",
        "![ROI and mROI by channel, Meridian vs PyMC-Marketing](../figures/meridian_vs_pymc_roi.png)",
        "",
        "## Do the business conclusions agree?",
        "",
        f"- **Ranking by ROI.** Meridian: {', '.join(rank_roi[0])}. PyMC-Marketing: {', '.join(rank_roi[1])}. "
        f"Spearman rank correlation {agreement['roi_rank_spearman']:.2f}.",
        f"- **Ranking by mROI.** Meridian: {', '.join(rank_mroi[0])}. PyMC-Marketing: {', '.join(rank_mroi[1])}. "
        f"Spearman rank correlation {agreement['mroi_rank_spearman']:.2f}.",
        f"- **Is every channel's next dollar below break-even (mROI < 1)?** Meridian: "
        f"{'yes' if agreement['all_mroi_below_1_meridian'] else 'no'}. PyMC-Marketing: "
        f"{'yes' if agreement['all_mroi_below_1_pymc'] else 'no'}.",
        f"- **PyMC-Marketing's mean ROI falls inside Meridian's {pct}% interval** for "
        f"{agreement['pymc_roi_mean_inside_meridian_interval']} of 5 channels.",
    ]
    ml, pl = mer["budget_level"], level

    def lost(s: dict) -> dict:  # a revenue change as a loss: negate and swap the interval ends
        return {"mean": -s["mean"], "ci_low": -s["ci_high"], "ci_high": -s["ci_low"]}

    lines += [
        f"- **Cut every paid channel by 10%** (saves {money(-pl['spend_change'])}). Revenue lost: Meridian "
        f"{ci_money(lost(ml['change_in_incremental_revenue']))}, "
        f"PyMC-Marketing {ci_money(lost(pl['change_in_incremental_revenue']))}. "
        f"Share of draws in which the revenue lost is smaller than the spend saved: Meridian "
        f"{ml['prob_revenue_change_smaller_than_spend_change']:.1%}, PyMC-Marketing "
        f"{pl['prob_revenue_change_smaller_than_spend_change']:.1%}.",
    ]
    mr = mer["reallocation"]
    rs = realloc_same
    lines.append(
        f"- **Move 10% of {m_src}'s budget to {m_tgt}** (the move in the Meridian report, {money(mr['spend_moved'])}). "
        f"Change in revenue: Meridian {ci_money(mr['change_in_incremental_revenue'])}, "
        f"P(gain) {mr['change_in_incremental_revenue']['prob_positive']:.1%}; PyMC-Marketing "
        f"{ci_money(rs['change_in_incremental_revenue'])}, P(gain) {rs['change_in_incremental_revenue']['prob_positive']:.1%}.")
    if realloc_own is not realloc_same:
        ro = realloc_own
        lines.append(
            f"- PyMC-Marketing's own lowest-mROI to highest-mROI move is {p_src} to {p_tgt}: "
            f"{ci_money(ro['change_in_incremental_revenue'])}, P(gain) {ro['change_in_incremental_revenue']['prob_positive']:.1%}.")
    lines += [
        "",
        "## Carryover (adstock half-life, weeks)",
        "",
        "| Channel | Meridian | PyMC-Marketing | Weight after 1 week: Meridian | PyMC-Marketing |",
        "|---|---:|---:|---:|---:|",
    ]
    for ch in pmm.CHANNELS:
        def hl(t):
            v = t.loc[ch, "half_life_weeks"]
            return "> 8" if pd.isna(v) else f"{v:.1f}"
        lines.append(f"| {ch} | {hl(mh)} | {hl(ph)} | {mh.loc[ch, 'weight_after_1_week']:.2f} | "
                     f"{ph.loc[ch, 'weight_after_1_week']:.2f} |")
    lo, hi = np.exp(config.ROI_PRIOR_MU + np.array([-1, 1]) * 1.6449 * config.ROI_PRIOR_SIGMA)
    lines += [
        "",
        "## The ROI prior in each model",
        "",
        f"Meridian: LogNormal(0.2, 0.9) for every paid channel, median 1.22, {pct}% interval [{lo:.2f}, {hi:.2f}].",
        "",
        f"PyMC-Marketing (implied by the effect priors; {fit.get('calibration_draws', pmm.N_CALIBRATION_DRAWS):,} "
        "prior draws per channel):",
        "",
        "| Channel | Prior ROI median | 5th percentile | 95th percentile | sd of log ROI |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in calibration.iterrows():
        lines.append(f"| {r['channel']} | {r['prior_roi_median']:.2f} | {r['prior_roi_p05']:.2f} | "
                     f"{r['prior_roi_p95']:.2f} | {r['prior_log_roi_sd']:.2f} |")
    lines += [
        "",
        "The medians and log-scale spreads match; the shapes differ a little (PyMC-Marketing's implied prior is "
        "not exactly log-normal).",
        "",
    ]
    (out / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
