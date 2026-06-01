from pathlib import Path

# ─── 路径 ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT
ARTIFACTS_DIR = ROOT / "artifacts"
EDA_DIR = ARTIFACTS_DIR / "eda"
OOF_DIR = ARTIFACTS_DIR / "oof"
PREDS_DIR = ARTIFACTS_DIR / "preds"
MODELS_DIR = ARTIFACTS_DIR / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

# ─── 目标与类别 ─────────────────────────────────────────
TARGET = "class"
CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ─── 特征列 ─────────────────────────────────────────────
ID_COL = "id"
NUMERIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]
MAG_COLS = ["u", "g", "r", "i", "z"]

# ─── 随机种子 & CV ───────────────────────────────────────
SEED = 42
N_FOLDS = 5
