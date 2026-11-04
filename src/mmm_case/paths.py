from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "geo_all_channels.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_FILE = ARTIFACTS / "model.binpb"
FIT_INFO = ROOT / "reports" / "fit_info.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
