"""
run_calibration.py

WHY THIS FILE EXISTS
---------------------
Thin entry point for calibration.py (Person 5's calibration pipeline),
both living in app/ml/. Run THIS file instead of
`python -m app.ml.calibration` directly.

Person 5's calibration.py defines its own classes (_LGBAdapter,
PlattScaling, IsotonicCalibration) and embeds instances of them directly
inside the saved .joblib artifact, as CalibratedChurnModel's base_model /
calibrator fields — that design is unchanged here.

The problem is purely about HOW that script gets invoked. Python's pickle
module records a class's location as whichever module name that class's
file had at definition time. If calibration.py is executed directly
(`python -m app.ml.calibration` or `python app/ml/calibration.py`),
Python sets its __name__ to "__main__" for that run, so _LGBAdapter /
PlattScaling / IsotonicCalibration get pickled as belonging to module
"__main__". Any OTHER process loading that artifact afterwards (the API,
a test, a one-off script) has no "__main__._LGBAdapter" to find, and
unpickling fails with AttributeError.

Running calibration.py through this wrapper instead means Python imports
it as the real module `app.ml.calibration` (never as `__main__`), so
every class it defines gets its correct, stable __module__ path
(`app.ml.calibration`), and the saved artifact loads correctly from
anywhere.

USAGE
-----
    python app/ml/run_calibration.py

(from the repo root, same convention as scripts/run_pipeline.py)
"""

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
