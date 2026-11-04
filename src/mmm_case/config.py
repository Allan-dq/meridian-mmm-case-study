"""Model and sampler settings, fixed before the first full fit.

The model specification follows Meridian's Getting Started notebook for this dataset:
LogNormal(0.2, 0.9) ROI prior on every paid channel and automatic knot selection for the
time effects. The sampler uses 4 chains (one per performance core is plenty on a laptop);
the notebook uses 10 on a GPU.
"""

ROI_PRIOR_MU = 0.2
ROI_PRIOR_SIGMA = 0.9
ENABLE_AKS = True

SAMPLER = {"n_chains": 4, "n_adapt": 1000, "n_burnin": 500, "n_keep": 1000, "seed": 0}
N_PRIOR_DRAWS = 500
QUICK_SAMPLER = {"n_chains": 2, "n_adapt": 50, "n_burnin": 50, "n_keep": 50, "seed": 0}
QUICK_PRIOR_DRAWS = 50

CONFIDENCE_LEVEL = 0.9          # credible intervals reported everywhere
REALLOCATION_SHARE = 0.10       # "move 10% of channel X's budget to channel Y"
RHAT_THRESHOLD = 1.2            # Meridian's ModelReviewer convergence threshold
