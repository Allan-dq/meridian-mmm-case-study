"""Static charts for the README (light surface; colors from a CVD-checked reference palette)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"       # categorical slot 1
ORANGE = "#eb6834"     # categorical slot 2
BLUE_BAND = "#cde2fb"  # blue ramp step 100
RED = "#e34948"        # diverging pole opposite blue

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "axes.axisbelow": True,
})


def roi_mroi(metrics: pd.DataFrame, path, confidence: float) -> None:
    df = metrics[~metrics["channel"].str.startswith("All")].sort_values("roi_mean")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(df) + 1.6))
    for name, col, off, color in [("ROI", "roi", 0.14, BLUE), ("mROI", "mroi", -0.14, ORANGE)]:
        lo, hi, mid = df[f"{col}_ci_low"], df[f"{col}_ci_high"], df[f"{col}_mean"]
        ax.hlines(y + off, lo, hi, color=color, linewidth=2)
        ax.plot(mid, y + off, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=f"{name} (mean, {int(confidence * 100)}% interval)")
    ax.axvline(1.0, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)), label="break-even (1.0)")
    ax.set_yticks(y, df["channel"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Incremental revenue per $1 of spend")
    ax.set_title("Return on ad spend by channel: average (ROI) vs next dollar (mROI)", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def response_curves(curves: pd.DataFrame, path) -> None:
    channels = sorted(curves["channel"].unique())
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.9), sharey=False)
    for ax, ch in zip(np.atleast_1d(axes), channels):
        g = curves[curves["channel"] == ch].sort_values("spend_multiplier")
        x = g["spend"] / 1e6
        ax.fill_between(x, g["ci_low"] / 1e6, g["ci_high"] / 1e6, color=BLUE_BAND, linewidth=0)
        ax.plot(x, g["mean"] / 1e6, color=BLUE, linewidth=2)
        cur = g[np.isclose(g["spend_multiplier"], 1.0)]
        ax.plot(cur["spend"] / 1e6, cur["mean"] / 1e6, "o", color=BLUE, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate("current", (cur["spend"].iloc[0] / 1e6, cur["mean"].iloc[0] / 1e6),
                    textcoords="offset points", xytext=(-4, 8), ha="right", fontsize=8, color=TEXT_2)
        ax.set_title(ch, loc="left")
        ax.set_xlabel("Spend ($M)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
    np.atleast_1d(axes)[0].set_ylabel("Incremental revenue ($M)")
    fig.suptitle("Response curves: 0x to 2x of historical spend (band = 90% interval)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def budget_shift(table: pd.DataFrame, path) -> None:
    df = table.sort_values("spend_change_pct")
    y = np.arange(len(df))
    pct = df["spend_change_pct"] * 100
    colors = [BLUE if v >= 0 else RED for v in pct]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(df) + 1.4))
    ax.barh(y, pct, color=colors, height=0.55, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, pct):
        ax.text(v + (1 if v >= 0 else -1), yi, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=TEXT)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, df["channel"])
    ax.grid(axis="y", visible=False)
    lim = max(35, float(np.abs(pct).max()) + 8)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
    ax.set_xlabel("Change in spend vs current plan (same total budget)")
    ax.set_title("Optimizer's reallocation (each channel bounded to +/-30%)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def model_comparison(meridian: pd.DataFrame, pymc: pd.DataFrame, path, confidence: float) -> None:
    """ROI and mROI by channel, Meridian vs PyMC-Marketing, each with its credible interval."""
    order = meridian[~meridian["channel"].str.startswith("All")].sort_values("roi_mean")["channel"].tolist()
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.6 * len(order) + 1.9), sharey=True)
    for ax, (col, title) in zip(axes, [("roi", "ROI: average return"), ("mroi", "mROI: next dollar")]):
        for name, df, off, color in [("Meridian", meridian, 0.14, BLUE), ("PyMC-Marketing", pymc, -0.14, ORANGE)]:
            d = df.set_index("channel").loc[order]
            ax.hlines(y + off, d[f"{col}_ci_low"], d[f"{col}_ci_high"], color=color, linewidth=2)
            ax.plot(d[f"{col}_mean"], y + off, "o", color=color, markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=name)
        ax.axvline(1.0, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Incremental revenue per $1 of spend")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
    axes[0].set_yticks(y, order)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Same data, two Bayesian MMMs (mean and {int(confidence * 100)}% interval; dashed line = break-even)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
