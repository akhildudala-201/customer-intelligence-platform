"""
conftest.py (app/ml/tests/explainibility_inference_tests/)

Ensures the repo root is on sys.path so `from app.ml... import` and
`from app... import` resolve regardless of which directory pytest is
invoked from -- mirrors the equivalent setup in the top-level
tests/conftest.py (owned by Feature Engineering), duplicated here rather
than shared because app/ml/tests/explainibility_inference_tests/ is not
a descendant of tests/, so pytest's automatic conftest.py inheritance
doesn't reach across that boundary. pytest still composes conftest.py
files within THIS subtree automatically
(app/ml/tests/explainibility_inference_tests/inference/conftest.py's
fixtures apply only to
app/ml/tests/explainibility_inference_tests/inference/, this file's
sys.path setup applies to everything under
app/ml/tests/explainibility_inference_tests/, api/ included).

Note: this only sets up sys.path -- it does NOT import anything from
app.ml.explainibility_inference. The source code this suite tests still
lives at app/ml/explainibility_inference/ (unchanged); only the tests
moved out to this shared app/ml/tests/ location, one subfolder per
ml-layer contributor.
"""

from pathlib import Path
import sys

# This file lives at app/ml/tests/explainibility_inference_tests/conftest.py
# -- same depth from the repo root as the old location
# (app/ml/explainibility_inference/tests/conftest.py) had, so parents[4]
# is still correct: explainibility_inference_tests -> tests -> ml -> app -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
