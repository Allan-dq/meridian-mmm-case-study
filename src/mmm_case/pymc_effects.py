"""Meridian's time effect as a PyMC-Marketing additive effect.

Meridian models a national time effect with knots: one value per knot, linearly interpolated
between neighbouring knots (`meridian.model.knots.l1_distance_weights`). `KnotTimeEffect` adds the
same term to a PyMC-Marketing MMM, with the knot positions Meridian chose for this dataset
(`reports/pymc/meridian_knots.json`), so the two models share their time-effect structure.

One difference is deliberate: each basis column is centred over the fitted weeks. The term then
averages zero over the fit period and does not compete with the geo intercepts for the overall
level (Meridian removes the same ambiguity by fixing one geo's intercept at zero). Because the
centred columns sum to zero, adding a constant to every knot value would not change the term, so
the knot values get a zero-sum prior: the data cannot inform that direction, and leaving it in only
slows the sampler down.

This module imports PyMC-Marketing at import time; it must be imported before `MMM.load` so that
the effect's class is registered for deserialization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
from pymc_marketing.mmm.additive_effect import MuEffect

from mmm_case.pymc_model import knot_weights


class KnotTimeEffect(MuEffect):
    """National time effect: knot values interpolated linearly over weeks, centred over the fit period."""

    knot_locations: list[int]
    start_date: str
    n_times: int
    prior_sigma: float = 0.5
    freq_days: int = 7
    prefix: str = "time_knots"
    date_dim_name: str = "date"

    def _basis(self, dates) -> np.ndarray:
        t = (pd.to_datetime(np.asarray(dates)) - pd.Timestamp(self.start_date)).days / self.freq_days
        knots = np.asarray(self.knot_locations, dtype=float)
        fit_mean = knot_weights(self.n_times, knots).mean(axis=1)            # (knot,)
        return knot_weights(np.asarray(t, dtype=float), knots).T - fit_mean   # (date, knot)

    def create_data(self, mmm) -> None:
        model = mmm.model
        model.add_coords({f"{self.prefix}_knot": np.arange(len(self.knot_locations))})
        pmd.Data(f"{self.prefix}_basis", self._basis(model.coords[self.date_dim_name]),
                 dims=(self.date_dim_name, f"{self.prefix}_knot"))

    def create_effect(self, mmm):
        model = mmm.model
        knot_dim = f"{self.prefix}_knot"
        values = pmd.ZeroSumNormal(f"{self.prefix}_values", sigma=self.prior_sigma,
                                   core_dims=knot_dim, dims=(knot_dim,))
        basis = model[f"{self.prefix}_basis"]
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution", (basis * values).sum(dim=f"{self.prefix}_knot")
        )

    def set_data(self, mmm, model: pm.Model, X) -> None:
        pm.set_data({f"{self.prefix}_basis": self._basis(model.coords[self.date_dim_name])}, model=model)
