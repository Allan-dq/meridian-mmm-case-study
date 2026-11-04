"""Build and fit the Meridian model."""

import numpy as np

from mmm_case import config


def build_model(input_data):
    import tensorflow_probability.substrates.jax as tfp_jax
    from meridian import constants
    from meridian.model import model, prior_distribution, spec

    prior = prior_distribution.PriorDistribution(
        roi_m=tfp_jax.distributions.LogNormal(
            np.float64(config.ROI_PRIOR_MU), np.float64(config.ROI_PRIOR_SIGMA), name=constants.ROI_M
        )
    )
    model_spec = spec.ModelSpec(prior=prior, enable_aks=config.ENABLE_AKS)
    return model.Meridian(input_data=input_data, model_spec=model_spec)


def fit(mmm, quick: bool = False):
    sampler = config.QUICK_SAMPLER if quick else config.SAMPLER
    n_prior = config.QUICK_PRIOR_DRAWS if quick else config.N_PRIOR_DRAWS
    mmm.sample_prior(n_prior, seed=sampler["seed"])
    mmm.sample_posterior(**sampler)
    return sampler, n_prior
