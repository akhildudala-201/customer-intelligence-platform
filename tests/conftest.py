from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "app" / "Features"
DATABASE_DIR = PROJECT_ROOT / "app" / "Database"
ML_DIR = PROJECT_ROOT / "app" / "ml"
ML_UPPER_DIR = PROJECT_ROOT / "app" / "ML"

for path in (str(PROJECT_ROOT), str(FEATURES_DIR), str(DATABASE_DIR), str(ML_DIR), str(ML_UPPER_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
