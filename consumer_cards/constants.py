"""The scientific configuration retained from the recorded October 2026 study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "reference" / "card_clients" / "historical"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
SOURCE_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
PROTOCOL_SHA256 = "7376e59998b3563fbddca17c4ab1cd9dde1e089118a62dcba42940aa126f1ce3"
SEED = 20261010
TARGET = "default payment next month"
EXCLUDED = ("EDUCATION", "ID", "MARRIAGE", "SEX")
FEATURES = (
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)
CATEGORICAL = tuple(name for name in FEATURES if name.startswith("PAY_") and not name.startswith("PAY_AMT"))
CONTINUOUS = tuple(name for name in FEATURES if name not in CATEGORICAL)
CAPACITIES = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 1.0)
MODEL_NAMES = ("scorecard", "challenger")
HISTORICAL_HASHES = {
    "PROTOCOL.json": PROTOCOL_SHA256,
    "summary.json": "1e09b502b89f65464cd97d95965f2f519c11233ac0d86c5d9ac873f61924c7c7",
    "partition_manifest.csv": "aea19b791d08c1c2c987efe404a77d4564ccd3f64bf3583c10d2ab98031600e2",
    "holdout_predictions.csv": "e2c70c6f144a17b068d8b05a21641e80c01350c638afdac524a84f091d320dea",
    "calibration_bins.csv": "fc032fd24968a4ae268bbec332ec3fc963981b947c5963f949cfbaf07007b4e8",
    "group_audit.csv": "b71f198730b79caa3f26f37c61ce12010b161efe11131cee16e8d5748fc7972b",
    "runtime.json": "485813981d1738b9c57b16be0fe9e257c89d9b3b67690951b3a12213ac7dec66",
    "source/benchmark_card_clients.py": "900d940fb663e4c103c0dbef0bf61a4294d617742741d060f46ed2179c0c792a",
    "source/src/woe.py": "c575e161e4f2e9ded98f0e14cda395413193438934f0c238cc4a00a805ab8aac",
    "source/requirements-card-clients.txt": "6d917faeaf19d53ebbbd47c17c43bc648f328f4f7705aa36fb1cfc864b7bc277",
}
