"""Numerical questionnaire classification using the user-selected Kaggle dataset."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "akashnath29-numeric-v1"
TARGET = "LUNG_CANCER"
BINARY_FEATURES = (
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
    "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN",
)
FEATURES = ("AGE", "GENDER", *BINARY_FEATURES)
LABELS = ("NO", "YES")
DATA = ROOT / "data/numerical/akashnath29"
REPORTS = ROOT / "reports/numerical"
ARTIFACTS = ROOT / "artifacts/numerical"
DATASET_URL = "https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset"
