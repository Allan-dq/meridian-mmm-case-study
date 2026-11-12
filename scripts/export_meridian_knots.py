"""Write the knot positions Meridian chose for its time effect, for the PyMC-Marketing comparison.

Run with the Meridian environment after `make fit`:

    python scripts/export_meridian_knots.py      # writes reports/pymc/meridian_knots.json

The PyMC-Marketing model reuses these positions so that both models have the same time effect.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmm_case import paths  # noqa: E402


def main() -> None:
    import numpy as np
    from meridian.schema.serde import meridian_serde

    mmm = meridian_serde.load_meridian(str(paths.MODEL_FILE))
    info = mmm.knot_info
    out = paths.REPORTS / "pymc" / "meridian_knots.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_times": int(mmm.n_times),
        "n_knots": int(info.n_knots),
        "knot_locations": [int(k) for k in np.asarray(info.knot_locations)],
        "source": "Meridian knot_info of artifacts/model.binpb (automatic knot selection)",
    }, indent=2) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
