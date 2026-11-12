"""Fit the PyMC-Marketing comparison model and save it.

Run with the PyMC environment (requirements-pymc.txt), not the Meridian one:

    .venv-pymc/bin/python scripts/fit_pymc.py            # full fit, ~minutes on a laptop
    .venv-pymc/bin/python scripts/fit_pymc.py --quick    # a few draws, to check the code path

Writes artifacts/pymc_model.nc (git-ignored), reports/pymc/prior_calibration.csv and
reports/pymc/fit_info.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmm_case import pymc_model as pmm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--centered", action="store_true",
                    help="centred parameterization of the geo-level effects (default: non-centred)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "pymc_model.nc"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "pymc"))
    args = ap.parse_args()

    import arviz as az
    import numpy as np
    import nutpie
    import pymc
    import pymc_marketing

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    frame = pmm.model_frame()
    n_cal = 4000 if args.quick else pmm.N_CALIBRATION_DRAWS
    calibration = pmm.calibrate_roi_prior(frame, n=n_cal)
    calibration.to_csv(reports / "prior_calibration.csv", index=False)

    mmm = pmm.build_mmm(frame, calibration, centered=args.centered)
    X, y = frame.drop(columns=[pmm.TARGET]), frame[pmm.TARGET]
    sampler = pmm.QUICK_SAMPLER if args.quick else pmm.SAMPLER
    t0 = time.time()
    idata = mmm.fit(X, y, nuts_sampler="nutpie", progressbar=False, **sampler)
    runtime = time.time() - t0

    fit = pmm.national_fit(idata, frame)
    names = [v.name for v in mmm.model.free_RVs]
    summ = az.summary(idata, var_names=names, kind="diagnostics")
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    pmm.drop_large_deterministics(mmm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mmm.save(args.out)

    info = {
        "quick": args.quick,
        "sampler": {**sampler, "nuts_sampler": "nutpie"},
        "runtime_seconds": round(runtime, 1),
        "divergences": divergences,
        "max_rhat": float(summ["r_hat"].max()),
        "max_rhat_parameter": str(summ["r_hat"].idxmax()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "national_fit": fit,
        "calibration_draws": n_cal,
        "geo_effects": "centred" if args.centered else "non-centred",
        "pymc_marketing": pymc_marketing.__version__,
        "pymc": pymc.__version__,
        "nutpie": nutpie.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "model_file": Path(args.out).name,
    }
    (reports / "fit_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
