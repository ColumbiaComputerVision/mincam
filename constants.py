from pathlib import Path

DATASET_PATH = Path("datasets").resolve()

DATA_PATH = Path("data").resolve()

MODEL_PATH = DATA_PATH / "models"
LOG_PATH = DATA_PATH / "logs"
EXP_CONFIGS_PATH = DATA_PATH / "exp_configs"

DIODE_DIRECTIVITY_PATH = DATA_PATH / "S9119-01-directivity.mat"

# Create directories if they do not already exist
MODEL_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)
EXP_CONFIGS_PATH.mkdir(exist_ok=True)
