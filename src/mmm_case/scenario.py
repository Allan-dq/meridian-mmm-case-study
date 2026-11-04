"""Budget reallocation scenarios on the model's own input data.

A channel's budget change is applied the way Meridian's budget optimizer applies it: every
geo-week of that channel's historical spend and impressions is multiplied by the same factor.
This keeps each channel's flighting pattern and its historical cost per impression ("constant
CPM"), which is the standard planning assumption.
"""

import numpy as np


def reallocate(media: np.ndarray, spend: np.ndarray, channels: list[str], source: str, target: str,
               share: float) -> tuple[np.ndarray, np.ndarray]:
    """Move `share` of `source`'s total spend to `target`, keeping total spend fixed.

    `media` and `spend` have shape (geo, time, channel). Returns new float copies.
    """
    if not 0 <= share <= 1:
        raise ValueError("share must be between 0 and 1")
    if source == target:
        raise ValueError("source and target must differ")
    s, t = channels.index(source), channels.index(target)
    media, spend = media.astype(float).copy(), spend.astype(float).copy()
    moved = share * spend[..., s].sum()
    target_total = spend[..., t].sum()
    if target_total <= 0:
        raise ValueError(f"{target} has no historical spend to scale")
    factors = np.ones(len(channels))
    factors[s] = 1 - share
    factors[t] = 1 + moved / target_total
    return media * factors, spend * factors
