"""The same dataset fitted with PyMC-Marketing, set up to mirror the Meridian model.

What is matched, and what cannot be, is listed in `reports/pymc/comparison.md`. In short:

- revenue (conversions x revenue per conversion) is the target, as in Meridian's ROI analysis;
- paid channels enter as spend; each channel's cost per impression is constant in this dataset,
  so spend and impressions differ only by a fixed factor per channel;
- geometric adstock over lags 0-8 (Meridian's `max_lag=8`), applied before saturation;
- Hill saturation with the slope fixed at 1 (Meridian's default), i.e. Michaelis-Menten;
- adstock rate and half-saturation point are national per channel, and the channel effect
  varies by geo around a national level (log-normal), as in Meridian;
- one noise scale shared by all geos (Meridian's default);
- the same national time effect: Meridian's 66 knots, linearly interpolated (pymc_effects.py);
- the prior on each paid channel's effect is chosen so that the prior on its ROI matches
  Meridian's LogNormal(0.2, 0.9). PyMC-Marketing has no ROI parameterization, so the match is
  made numerically by `calibrate_roi_prior`.

Everything here imports PyMC-Marketing lazily, so the module can be imported (and its pure
NumPy helpers tested) without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_case import config
from mmm_case.data import CONTROLS, NON_MEDIA_TREATMENTS, ORGANIC_MEDIA, PAID_CHANNELS, load_frame

DATE, GEO, TARGET = "time", "geo", "revenue"
CHANNELS = PAID_CHANNELS + ORGANIC_MEDIA          # model coordinate names
CONTROL_COLUMNS = CONTROLS + NON_MEDIA_TREATMENTS  # the promotion enters as a control
L_MAX = 9                                          # PyMC-Marketing counts lag 0, so 9 = lags 0..8

# Meridian priors this model mirrors (see config.py and Meridian's default PriorDistribution).
ROI_LOG_MU, ROI_LOG_SIGMA = config.ROI_PRIOR_MU, config.ROI_PRIOR_SIGMA
EC_LOC, EC_SCALE, EC_LOW, EC_HIGH = 0.8, 0.8, 0.1, 10.0  # Meridian ec_m, in units of median media
GEO_SPREAD_SCALE = 1.0                                     # Meridian eta_m ~ HalfNormal(1)
KNOT_PRIOR_SIGMA = 0.5   # vague: ~3 within-geo sds of the max-scaled target (Meridian: Normal(0, 5))

SAMPLER = {"chains": 4, "tune": 1500, "draws": 1000, "target_accept": 0.99, "random_seed": 0}
QUICK_SAMPLER = {"chains": 2, "tune": 100, "draws": 100, "random_seed": 0}
# Posterior deterministics with a value per geo-week; recomputed from the parameters when needed.
# Dropping them before saving keeps the model file small (about 2 GB with them at 4,000 draws).
LARGE_DETERMINISTICS = ["mu", "channel_contribution", "control_contribution", "intercept_contribution",
                        "total_media_contribution_original_scale"]
N_CALIBRATION_DRAWS = 20_000


def model_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per geo-week: date, geo, revenue, channel inputs (spend for paid), controls."""
    df = load_frame() if df is None else df
    out = pd.DataFrame({
        DATE: pd.to_datetime(df["time"]),
        GEO: df["geo"].astype(str),
        TARGET: df["conversions"] * df["revenue_per_conversion"],
    })
    for ch in PAID_CHANNELS:
        out[ch] = df[f"{ch}_spend"].astype(float)
    for ch in ORGANIC_MEDIA:
        out[ch] = df[f"{ch}_impression"].astype(float)
    for c in CONTROL_COLUMNS:
        out[c] = df[c].astype(float)
    return out.sort_values([DATE, GEO]).reset_index(drop=True)


def panel(frame: pd.DataFrame, column: str) -> np.ndarray:
    """(geo, week) array of one column, geos in sorted order."""
    return frame.pivot(index=GEO, columns=DATE, values=column).sort_index().to_numpy(float)


def geometric_adstock(x: np.ndarray, alpha: np.ndarray, l_max: int = L_MAX) -> np.ndarray:
    """Normalized geometric adstock along the last axis, for a batch of decay rates.

    `x` is (geo, week); `alpha` is (n,). Returns (n, geo, week). Same convention as
    PyMC-Marketing's `GeometricAdstock(normalize=True)`: weights alpha**l for l = 0..l_max-1,
    divided by their sum.
    """
    alpha = np.asarray(alpha, float)[:, None]
    w = alpha ** np.arange(l_max)[None, :]
    w = w / w.sum(axis=1, keepdims=True)
    out = np.zeros((len(alpha), *x.shape))
    for lag in range(l_max):
        shifted = np.zeros_like(x)
        shifted[:, lag:] = x[:, : x.shape[1] - lag]
        out += w[:, lag][:, None, None] * shifted[None]
    return out


def median_nonzero_ratio(x_scaled: np.ndarray) -> float:
    """Median of a channel's non-zero max-scaled values, across all geo-weeks.

    Meridian measures the half-saturation point in units of the channel's median non-zero media;
    PyMC-Marketing's inputs are divided by each geo's maximum. This ratio converts one to the other.
    """
    nz = x_scaled[x_scaled > 0]
    return float(np.median(nz)) if nz.size else 1.0


def half_saturation_prior_params(ratio: float) -> dict:
    """Meridian's ec_m ~ TruncatedNormal(0.8, 0.8, 0.1, 10), rescaled to max-scaled media."""
    return {"mu": EC_LOC * ratio, "sigma": EC_SCALE * ratio, "lower": EC_LOW * ratio, "upper": EC_HIGH * ratio}


def _truncnorm_draws(rng, mu, sigma, lower, upper, n):
    from scipy.stats import truncnorm

    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)


def implied_log_roi(spend: np.ndarray, target_max: np.ndarray, log_level_sd: float,
                    n: int = N_CALIBRATION_DRAWS, seed: int = 0) -> np.ndarray:
    """Prior draws of log ROI for one channel when the national log effect level is ~ N(0, sd).

    The effect of the channel in geo g is A_g = exp(mu + sigma * z_g) on the max-scaled target, with
    mu ~ N(0, log_level_sd), sigma ~ HalfNormal(1), z_g ~ N(0, 1); decay ~ Uniform(0, 1) and the
    half-saturation point follows `half_saturation_prior_params`. ROI = total contribution in
    revenue units / total spend. Adding a constant c to mu adds c to log ROI, which is how
    `calibrate_roi_prior` centres the prior.
    """
    rng = np.random.default_rng(seed)
    gmax = spend.max(axis=1, keepdims=True)
    x = np.divide(spend, gmax, out=np.zeros_like(spend), where=gmax > 0)
    lam = _truncnorm_draws(rng, **half_saturation_prior_params(median_nonzero_ratio(x)), n=n)
    decay = rng.uniform(0, 1, n)
    mu = rng.normal(0, log_level_sd, n)
    sigma = np.abs(rng.normal(0, GEO_SPREAD_SCALE, n))
    z = rng.normal(size=(n, spend.shape[0]))
    a_g = np.exp(mu[:, None] + sigma[:, None] * z)                     # (n, geo)
    total = np.empty(n)
    for start in range(0, n, 2000):                                    # chunked: (chunk, geo, week)
        sl = slice(start, start + 2000)
        s = geometric_adstock(x, decay[sl])
        h = (s / (lam[sl, None, None] + s)).sum(axis=2)                # (chunk, geo)
        total[sl] = (a_g[sl] * h * target_max[None, :]).sum(axis=1)
    return np.log(total / spend.sum())


def calibrate_roi_prior(frame: pd.DataFrame, n: int = N_CALIBRATION_DRAWS, seed: int = 0) -> pd.DataFrame:
    """Per paid channel: centre and spread of the national log-effect prior that give an ROI prior
    with Meridian's median exp(0.2) and, as closely as possible, its log-scale sd of 0.9.

    The decay, saturation and geo-spread priors already make log ROI vary; the national level adds
    the rest. If they alone exceed 0.9, the level's sd is set to a small floor and the achieved sd
    is reported, so the mismatch is visible rather than hidden.
    """
    target_max = panel(frame, TARGET).max(axis=1)
    rows = []
    for ch in PAID_CHANNELS:
        spend = panel(frame, ch)
        base = implied_log_roi(spend, target_max, 0.0, n=n, seed=seed)
        rest_var = ROI_LOG_SIGMA**2 - base.var()
        level_sd = float(np.sqrt(rest_var)) if rest_var > 0.05**2 else 0.05
        draws = implied_log_roi(spend, target_max, level_sd, n=n, seed=seed)
        centre = ROI_LOG_MU - float(np.median(draws))
        rows.append({
            "channel": ch,
            "level_mu": centre,
            "level_sd": level_sd,
            "half_saturation_ratio": median_nonzero_ratio(spend / np.maximum(spend.max(axis=1, keepdims=True), 1e-300)),
            "prior_log_roi_sd_from_shape_only": float(base.std()),
            "prior_log_roi_sd": float(draws.std()),
            "prior_roi_median": float(np.exp(np.median(draws) + centre)),
            "prior_roi_p05": float(np.exp(np.quantile(draws, 0.05) + centre)),
            "prior_roi_p95": float(np.exp(np.quantile(draws, 0.95) + centre)),
        })
    return pd.DataFrame(rows)


def knot_weights(times, knot_locations) -> np.ndarray:
    """(n_knots, n_times) linear-interpolation weights between neighbouring knots.

    Same rule as Meridian's `knots.l1_distance_weights`: a time on a knot puts all its weight on
    that knot, a time between two knots splits it in inverse proportion to the distances, and a
    time outside the knots puts all its weight on the nearest end knot. `times` is either a count
    (0..n-1) or an array of (possibly fractional) week indices.
    """
    t = np.arange(times, dtype=float) if np.isscalar(times) else np.asarray(times, dtype=float)
    k = np.asarray(knot_locations, dtype=float)
    tc = np.clip(t, k[0], k[-1])
    right = np.clip(np.searchsorted(k, tc, side="right"), 1, len(k) - 1)
    left = right - 1
    frac = (tc - k[left]) / (k[right] - k[left])
    w = np.zeros((len(k), len(t)))
    cols = np.arange(len(t))
    w[left, cols] += 1 - frac
    w[right, cols] += frac
    return w


def load_meridian_knots(path=None) -> list[int]:
    """Knot positions Meridian chose for this dataset (written by scripts/export_meridian_knots.py)."""
    import json

    from mmm_case import paths

    path = path or paths.REPORTS / "pymc" / "meridian_knots.json"
    return json.loads(open(path).read())["knot_locations"]


def national_fit(idata, frame: pd.DataFrame) -> dict:
    """In-sample fit of national weekly revenue, from the posterior mean of the expected value.

    Same metrics as Meridian's `predictive_accuracy` at national level: R-squared, MAPE, wMAPE.
    """
    scale = idata.constant_data["target_scale"]
    expected = (idata.posterior["mu"] * scale).sum(GEO).mean(("chain", "draw"))
    obs = frame.groupby(DATE)[TARGET].sum()
    pred = expected.to_series().reindex(obs.index)
    err = pred - obs
    return {
        "R_Squared": float(1 - (err**2).sum() / ((obs - obs.mean()) ** 2).sum()),
        "MAPE": float((err.abs() / obs).mean()),
        "wMAPE": float(err.abs().sum() / obs.sum()),
    }


def drop_large_deterministics(mmm) -> None:
    """Remove per-geo-week deterministics from the posterior before saving (see LARGE_DETERMINISTICS)."""
    post = mmm.idata["posterior"].to_dataset()
    free = {rv.name for rv in mmm.model.free_RVs}
    drop = [v for v in LARGE_DETERMINISTICS if v in post and v not in free and "date" in post[v].dims]
    mmm.idata["posterior"] = post.drop_vars(drop)


def build_mmm(frame: pd.DataFrame, calibration: pd.DataFrame, knots: list[int] | None = None,
              centered: bool = False):
    """PyMC-Marketing MMM with the structure and priors described in the module docstring.

    `centered` switches the geo-level effects (channel effects and control coefficients) between
    the centred and non-centred parameterizations; the model is the same, only the sampler's
    geometry changes.
    """
    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import (
        MMM,
        DataDerivedScaling,
        GeometricAdstock,
        MichaelisMentenSaturation,
    )

    from mmm_case.pymc_effects import KnotTimeEffect

    cal = calibration.set_index("channel")
    # Organic channel: no spend, so no ROI to match; it gets the paid channels' average level.
    level_mu = np.array([*cal.loc[PAID_CHANNELS, "level_mu"], cal["level_mu"].mean()])
    level_sd = np.array([*cal.loc[PAID_CHANNELS, "level_sd"], 1.0])
    ratios = []
    for ch in CHANNELS:
        x = panel(frame, ch)
        ratios.append(median_nonzero_ratio(x / np.maximum(x.max(axis=1, keepdims=True), 1e-300)))
    hs = [half_saturation_prior_params(r) for r in ratios]

    adstock = GeometricAdstock(
        l_max=L_MAX,
        priors={"alpha": Prior("Beta", alpha=1, beta=1, dims="channel")},   # Meridian: Uniform(0, 1)
    )
    saturation = MichaelisMentenSaturation(priors={
        # effect level A[geo, channel] = exp(level[channel] + spread[channel] * z[geo, channel])
        "alpha": Prior(
            "Normal",
            mu=Prior("Normal", mu=level_mu, sigma=level_sd, dims="channel"),
            sigma=Prior("HalfNormal", sigma=GEO_SPREAD_SCALE, dims="channel"),
            dims=(GEO, "channel"),
            centered=centered,
            transform="exp",
        ),
        "lam": Prior(
            "TruncatedNormal",
            mu=np.array([h["mu"] for h in hs]),
            sigma=np.array([h["sigma"] for h in hs]),
            lower=np.array([h["lower"] for h in hs]),
            upper=np.array([h["upper"] for h in hs]),
            dims="channel",
        ),
    })
    model_config = {
        "intercept": Prior("Normal", mu=0, sigma=2, dims=(GEO,)),
        "gamma_control": Prior(
            "Normal",
            mu=Prior("Normal", mu=0, sigma=2, dims="control"),
            sigma=Prior("HalfNormal", sigma=1, dims="control"),
            dims=(GEO, "control"),
            centered=centered,
        ),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2), dims=("date", GEO)),
    }
    # Scale each geo by its own maximum (reduce over dates only). PyMC-Marketing's default for a
    # geo model reduces over geos too, i.e. one national maximum; per-geo scaling is the closer
    # analogue of Meridian's per-capita scaling, and it is what `calibrate_roi_prior` assumes.
    per_geo_max = DataDerivedScaling(method="max", dims=())
    mmm = MMM(
        date_column=DATE,
        channel_columns=CHANNELS,
        target_column=TARGET,
        control_columns=CONTROL_COLUMNS,
        dims=(GEO,),
        adstock=adstock,
        saturation=saturation,
        model_config=model_config,
        scaling={"target": per_geo_max, "channel": per_geo_max},
    )
    # National time effect with Meridian's knots (see pymc_effects.KnotTimeEffect).
    mmm.add_mu_effect(KnotTimeEffect(
        knot_locations=list(knots if knots is not None else load_meridian_knots()),
        start_date=str(frame[DATE].min().date()),
        n_times=int(frame[DATE].nunique()),
        prior_sigma=KNOT_PRIOR_SIGMA,
    ))
    return mmm
