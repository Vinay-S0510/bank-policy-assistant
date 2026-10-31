from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CHUNK_SIZE = 1_100
CHUNK_OVERLAP = 160
TOP_K = 5
