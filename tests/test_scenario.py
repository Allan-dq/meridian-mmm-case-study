import numpy as np
import pytest

from mmm_case.scenario import reallocate

CH = ["A", "B", "C"]


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    spend = rng.uniform(0, 100, size=(4, 10, 3))
    media = spend * np.array([10.0, 20.0, 5.0]) * rng.uniform(0.8, 1.2, size=spend.shape)
    return media, spend


def test_total_spend_is_conserved():
    media, spend = _arrays()
    _, new_spend = reallocate(media, spend, CH, "A", "B", 0.1)
    assert new_spend.sum() == pytest.approx(spend.sum())


def test_moves_exactly_the_share_of_source_spend():
    media, spend = _arrays()
    _, new_spend = reallocate(media, spend, CH, "A", "C", 0.25)
    assert new_spend[..., 0].sum() == pytest.approx(0.75 * spend[..., 0].sum())
    assert new_spend[..., 2].sum() - spend[..., 2].sum() == pytest.approx(0.25 * spend[..., 0].sum())
    np.testing.assert_allclose(new_spend[..., 1], spend[..., 1])


def test_media_keeps_each_channels_cost_per_impression():
    media, spend = _arrays()
    new_media, new_spend = reallocate(media, spend, CH, "B", "A", 0.1)
    for c in range(3):
        before = spend[..., c].sum() / media[..., c].sum()
        after = new_spend[..., c].sum() / new_media[..., c].sum()
        assert after == pytest.approx(before)


def test_flighting_pattern_is_kept():
    media, spend = _arrays()
    new_media, _ = reallocate(media, spend, CH, "A", "B", 0.1)
    ratio = new_media[..., 1] / media[..., 1]
    np.testing.assert_allclose(ratio, ratio.flat[0])


def test_zero_share_changes_nothing_and_inputs_are_not_mutated():
    media, spend = _arrays()
    m0, s0 = media.copy(), spend.copy()
    new_media, new_spend = reallocate(media, spend, CH, "A", "B", 0.0)
    np.testing.assert_allclose(new_media, media)
    np.testing.assert_allclose(new_spend, spend)
    np.testing.assert_array_equal(media, m0)
    np.testing.assert_array_equal(spend, s0)


def test_media_with_extra_lag_periods_is_supported():
    media, spend = _arrays()
    media_with_lags = np.concatenate([media[:, :3], media], axis=1)  # Meridian allows extra media weeks
    new_media, _ = reallocate(media_with_lags, spend, CH, "A", "B", 0.1)
    assert new_media.shape == media_with_lags.shape


@pytest.mark.parametrize("share", [-0.1, 1.5])
def test_rejects_invalid_share(share):
    media, spend = _arrays()
    with pytest.raises(ValueError):
        reallocate(media, spend, CH, "A", "B", share)


def test_rejects_same_channel():
    media, spend = _arrays()
    with pytest.raises(ValueError):
        reallocate(media, spend, CH, "A", "A", 0.1)
