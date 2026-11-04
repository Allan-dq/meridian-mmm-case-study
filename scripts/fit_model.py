"""Fit the model and save it with Meridian's serde format (artifacts/model.binpb).

    python scripts/fit_model.py          # full fit (about 16 minutes on an Apple M1 Pro CPU)
    python scripts/fit_model.py --quick  # a few draws, for smoke tests only
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm_case import data, model, paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny sampler settings, for smoke tests")
    parser.add_argument("--out", default=str(paths.MODEL_FILE))
    args = parser.parse_args()

    import jax
    import meridian
    from meridian.schema.serde import meridian_serde

    t0 = time.time()
    mmm = model.build_model(data.build_input_data())
    sampler, n_prior = model.fit(mmm, quick=args.quick)
    runtime = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meridian_serde.save_meridian(mmm, str(out))
    info = {
        "quick": args.quick,
        "sampler": sampler,
        "n_prior_draws": n_prior,
        "runtime_seconds": round(runtime, 1),
        "meridian": meridian.__version__,
        "jax": jax.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "model_file": out.name,
    }
    if not args.quick:
        paths.FIT_INFO.parent.mkdir(parents=True, exist_ok=True)
        paths.FIT_INFO.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
