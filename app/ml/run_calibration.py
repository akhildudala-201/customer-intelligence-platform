from __future__ import annotations

import sys
from pathlib import Path

# This file lives at app/ml/run_calibration.py, so parents[0]=app/ml,
# parents[1]=app, parents[2]=the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.calibration import main  # noqa: E402 — import after sys.path setup

if __name__ == "__main__":
    main()
