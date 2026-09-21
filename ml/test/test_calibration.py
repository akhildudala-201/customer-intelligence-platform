"""Unit tests for app/ml/calibration.py.

Run from the project root with either:
    python -m unittest tests.test_calibration -v
    pytest tests/test_calibration.py -v

Design notes
------------
* Tests are hermetic: no database, no real LightGBM, no dependency on the
  behaviour of Person 4's metrics module. Anything that talks to those
  (find_optimal_threshold, calculate_metrics, load_model_ready_data,
  log_experiment) is mocked where it matters.
* All file output (models, reports, plots) goes to a temporary directory by
  patching calibration.CONFIG.
* If the sibling modules that calibration.py imports at load time
  (app.ml.metrics, app.ml.experiment_logger, app.ml.logistic_regression)
  can't be imported in the test environment, lightweight stubs are
  registered so the module under test can still be imported.
"""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change this if your module lives under a different name/path.
MODULE_PATH = "app.ml.calibration"


def _stub_missing_dependencies() -> None:
    """Register stub modules for calibration.py's import-time dependencies
    ONLY if the real ones can't be imported."""
    stubs = {
        "app.ml.metrics": {
            "calculate_metrics": lambda **kw: {},
            "find_optimal_threshold": lambda y, p, metric="f1": (0.5, 0.0),
            "format_metrics_summary": lambda m, title="": str(m),
        },
        "app.ml.experiment_logger": {"log_experiment": lambda **kw: None},
        "app.ml.logistic_regression": {"load_model_ready_data": lambda: None},
    }
    for name, attrs in stubs.items():
        try:
            importlib.import_module(name)
        except Exception:
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            sys.modules[name] = mod


_stub_missing_dependencies()
cal = importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------------
# Test helpers / fakes (module-level so joblib can pickle them)
# ---------------------------------------------------------------------------

class ProbColumnModel:
    """Fake base model: its 'probability' is just the value in column 'p'."""

    feature_cols = ["p"]
    positive_class = 1

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            return X["p"].to_numpy(dtype=float)
        return np.asarray(X, dtype=float).ravel()


class StubSkModel:
    """Fake sklearn-style estimator returning a (n, 2) predict_proba.

    Column 1 is the first input feature; column 0 is its complement.
    Records the last X it was called with.
    """

    def __init__(self):
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = np.asarray(X)
        p = self.last_X[:, 0].astype(float)
        return np.column_stack([1.0 - p, p])


class HalvingCalibrator:
    """Trivial calibrator so CalibratedChurnModel can be tested in isolation."""

    def predict_proba(self, probs):
        return np.asarray(probs, dtype=float) * 0.5


def make_overconfident_data(n=4000, seed=0):
    """Synthetic data whose raw scores are systematically over-confident.

    True churn probability is uniform; the 'model' outputs sigmoid(3*logit(p)),
    i.e. pushed towards 0/1, so a calibrator has real work to do.
    """
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.98, n)
    y = rng.binomial(1, true_p)
    logit = np.log(true_p / (1 - true_p))
    raw = 1.0 / (1.0 + np.exp(-3.0 * logit))
    return pd.DataFrame({"p": raw}), pd.Series(y, name="churn_label")


class QuietTestCase(unittest.TestCase):
    """Base class: silences log_message/print in the module under test and
    provides a per-test temp directory."""

    def setUp(self):
        log_patcher = mock.patch.object(cal, "log_message")
        self.log = log_patcher.start()
        self.addCleanup(log_patcher.stop)

        print_patcher = mock.patch.object(cal, "print", create=True)
        self.print_mock = print_patcher.start()
        self.addCleanup(print_patcher.stop)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def use_tmp_output_dirs(self):
        model_dir = self.tmp / "models"
        report_dir = self.tmp / "reports"
        model_dir.mkdir()
        report_dir.mkdir()
        patcher = mock.patch.dict(
            cal.CONFIG,
            {
                "OUTPUT_MODEL_DIR": str(model_dir) + "/",
                "OUTPUT_REPORT_DIR": str(report_dir) + "/",
                "TIMESTAMP": "20240101_000000",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return model_dir, report_dir

    def logged(self, level):
        """All messages logged at the given level."""
        out = []
        for call in self.log.call_args_list:
            lvl = call.args[1] if len(call.args) > 1 else call.kwargs.get("level", "INFO")
            if lvl == level:
                out.append(call.args[0])
        return out


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

class TestExpectedCalibrationError(unittest.TestCase):
    def test_perfectly_calibrated_bin_has_zero_error(self):
        # 4 samples predicted 0.25, exactly 1 positive -> observed rate 0.25
        ece, mce = cal.expected_calibration_error([1, 0, 0, 0], [0.25] * 4)
        self.assertEqual((ece, mce), (0.0, 0.0))

    def test_single_bin_gap(self):
        # predicted 0.85, observed 0.5 -> gap 0.35
        y = [1] * 5 + [0] * 5
        ece, mce = cal.expected_calibration_error(y, [0.85] * 10)
        self.assertAlmostEqual(ece, 0.35, places=4)
        self.assertAlmostEqual(mce, 0.35, places=4)

    def test_two_symmetric_bins(self):
        y = [0] * 5 + [1] * 5
        p = [0.05] * 5 + [0.95] * 5
        ece, mce = cal.expected_calibration_error(y, p)
        self.assertAlmostEqual(ece, 0.05, places=4)
        self.assertAlmostEqual(mce, 0.05, places=4)

    def test_ece_is_weighted_by_bin_size_and_mce_is_worst_gap(self):
        # 8 samples @0.05 with 1 positive (gap .075); 2 samples @0.95, 0 positives (gap .95)
        y = [1] + [0] * 7 + [0, 0]
        p = [0.05] * 8 + [0.95] * 2
        ece, mce = cal.expected_calibration_error(y, p)
        self.assertAlmostEqual(ece, 0.8 * 0.075 + 0.2 * 0.95, places=4)  # 0.25
        self.assertAlmostEqual(mce, 0.95, places=4)
        self.assertGreaterEqual(mce, ece)

    def test_probability_of_exactly_one_is_counted_in_last_bin(self):
        # If the last bin weren't right-inclusive, these samples would be
        # dropped and ECE would (wrongly) be 0.
        ece, mce = cal.expected_calibration_error([1, 1, 1, 0], [1.0] * 4)
        self.assertAlmostEqual(ece, 0.25, places=4)
        self.assertAlmostEqual(mce, 0.25, places=4)

    def test_probability_of_exactly_zero_is_counted_in_first_bin(self):
        ece, _ = cal.expected_calibration_error([0, 0, 0, 1], [0.0] * 4)
        self.assertAlmostEqual(ece, 0.25, places=4)

    def test_n_bins_changes_resolution(self):
        y, p = [0, 1], [0.2, 0.8]
        ece_fine, _ = cal.expected_calibration_error(y, p, n_bins=10)
        ece_coarse, _ = cal.expected_calibration_error(y, p, n_bins=1)
        self.assertAlmostEqual(ece_fine, 0.2, places=4)
        self.assertAlmostEqual(ece_coarse, 0.0, places=4)  # miscalibration averages out

    def test_accepts_lists_and_pandas(self):
        y = pd.Series([0, 1, 1, 0])
        p = pd.Series([0.1, 0.9, 0.8, 0.2])
        ece_series = cal.expected_calibration_error(y, p)
        ece_list = cal.expected_calibration_error(list(y), list(p))
        self.assertEqual(ece_series, ece_list)

    def test_values_are_rounded_to_four_decimals(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 200)
        y = rng.binomial(1, 0.5, 200)
        for value in cal.expected_calibration_error(y, p):
            self.assertEqual(value, round(value, 4))


# ---------------------------------------------------------------------------
# calibration_metrics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics(unittest.TestCase):
    def test_returns_expected_keys(self):
        m = cal.calibration_metrics([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
        self.assertEqual(set(m), {"brier_score", "log_loss", "ece", "mce"})

    def test_perfect_predictions(self):
        m = cal.calibration_metrics([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(m["brier_score"], 0.0)
        self.assertEqual(m["log_loss"], 0.0)
        self.assertEqual(m["ece"], 0.0)

    def test_log_loss_is_finite_when_probabilities_are_exactly_0_or_1(self):
        # Wrong-but-certain predictions would give inf without clipping.
        m = cal.calibration_metrics([1, 0], [0.0, 1.0])
        self.assertTrue(np.isfinite(m["log_loss"]))
        self.assertGreater(m["log_loss"], 10)

    def test_constant_half_probability(self):
        m = cal.calibration_metrics([0, 1, 0, 1], [0.5] * 4)
        self.assertAlmostEqual(m["brier_score"], 0.25, places=4)
        self.assertAlmostEqual(m["log_loss"], 0.6931, places=4)

    def test_brier_matches_sklearn(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1, 500)
        y = rng.binomial(1, p)
        m = cal.calibration_metrics(y, p)
        self.assertAlmostEqual(m["brier_score"], round(brier_score_loss(y, p), 4), places=4)

    def test_worse_probabilities_score_worse(self):
        y = np.array([0, 0, 1, 1])
        good = cal.calibration_metrics(y, [0.1, 0.2, 0.8, 0.9])
        bad = cal.calibration_metrics(y, [0.9, 0.8, 0.2, 0.1])
        for key in ("brier_score", "log_loss", "ece"):
            self.assertLess(good[key], bad[key])


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

class TestPlattScaling(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            cal.PlattScaling().predict_proba([0.1, 0.9])

    def test_fit_returns_self_and_sets_flag(self):
        platt = cal.PlattScaling()
        self.assertFalse(platt.is_fitted_)
        self.assertIs(platt.fit([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1]), platt)
        self.assertTrue(platt.is_fitted_)

    def test_output_is_1d_probability_array(self):
        X, y = make_overconfident_data(n=500)
        out = cal.PlattScaling().fit(X["p"], y).predict_proba(X["p"])
        self.assertEqual(out.shape, (500,))
        self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_preserves_ranking(self):
        # Sigmoid scaling is monotonic in its input (positive relationship here).
        X, y = make_overconfident_data(n=1000)
        platt = cal.PlattScaling().fit(X["p"], y)
        grid = np.linspace(0.01, 0.99, 50)
        out = platt.predict_proba(grid)
        self.assertTrue((np.diff(out) >= 0).all())

    def test_accepts_plain_lists(self):
        platt = cal.PlattScaling().fit([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
        self.assertEqual(platt.predict_proba([0.2, 0.8]).shape, (2,))

    def test_improves_brier_on_overconfident_scores(self):
        X, y = make_overconfident_data()
        raw = X["p"].to_numpy()
        calibrated = cal.PlattScaling().fit(raw, y).predict_proba(raw)
        self.assertLess(brier_score_loss(y, calibrated), brier_score_loss(y, raw))


class TestIsotonicCalibration(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            cal.IsotonicCalibration().predict_proba([0.1, 0.9])

    def test_fit_returns_self_and_sets_flag(self):
        iso = cal.IsotonicCalibration()
        self.assertIs(iso.fit([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1]), iso)
        self.assertTrue(iso.is_fitted_)

    def test_perfectly_separable_data_maps_to_labels(self):
        iso = cal.IsotonicCalibration().fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        np.testing.assert_allclose(iso.predict_proba([0.1, 0.2, 0.8, 0.9]), [0, 0, 1, 1])

    def test_output_is_monotonic_non_decreasing(self):
        X, y = make_overconfident_data(n=1000)
        iso = cal.IsotonicCalibration().fit(X["p"], y)
        out = iso.predict_proba(np.linspace(0, 1, 200))
        self.assertTrue((np.diff(out) >= -1e-12).all())

    def test_out_of_range_inputs_are_clipped_not_nan(self):
        iso = cal.IsotonicCalibration().fit([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
        out = iso.predict_proba([-5.0, 5.0])
        self.assertFalse(np.isnan(out).any())
        self.assertEqual(out[0], iso.predict_proba([0.2])[0])
        self.assertEqual(out[1], iso.predict_proba([0.8])[0])

    def test_output_within_unit_interval(self):
        X, y = make_overconfident_data(n=500)
        out = cal.IsotonicCalibration().fit(X["p"], y).predict_proba(X["p"])
        self.assertTrue(((out >= 0) & (out <= 1)).all())


# ---------------------------------------------------------------------------
# CalibratedChurnModel
# ---------------------------------------------------------------------------

class TestCalibratedChurnModel(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"p": [0.2, 0.5, 0.8]})

    def test_predict_proba_without_calibrator_returns_raw(self):
        model = cal.CalibratedChurnModel(ProbColumnModel())
        np.testing.assert_allclose(model.predict_proba(self.X), [0.2, 0.5, 0.8])

    def test_predict_proba_applies_calibrator(self):
        model = cal.CalibratedChurnModel(ProbColumnModel(), calibrator=HalvingCalibrator())
        np.testing.assert_allclose(model.raw_predict_proba(self.X), [0.2, 0.5, 0.8])
        np.testing.assert_allclose(model.predict_proba(self.X), [0.1, 0.25, 0.4])

    def test_predict_threshold_is_inclusive(self):
        model = cal.CalibratedChurnModel(ProbColumnModel(), threshold=0.5)
        np.testing.assert_array_equal(model.predict(self.X), [0, 1, 1])

    def test_predict_uses_calibrated_probs_against_threshold(self):
        model = cal.CalibratedChurnModel(
            ProbColumnModel(), calibrator=HalvingCalibrator(), threshold=0.3
        )
        # calibrated = [0.1, 0.25, 0.4] -> only the last exceeds 0.3
        np.testing.assert_array_equal(model.predict(self.X), [0, 0, 1])

    def test_threshold_override_takes_precedence(self):
        model = cal.CalibratedChurnModel(ProbColumnModel(), threshold=0.5)
        np.testing.assert_array_equal(model.predict(self.X, threshold=0.9), [0, 0, 0])

    def test_threshold_override_of_zero_is_respected(self):
        # 0.0 is falsy; make sure it isn't mistaken for "no override".
        model = cal.CalibratedChurnModel(ProbColumnModel(), threshold=0.5)
        np.testing.assert_array_equal(model.predict(self.X, threshold=0.0), [1, 1, 1])

    def test_predict_returns_ints(self):
        preds = cal.CalibratedChurnModel(ProbColumnModel()).predict(self.X)
        self.assertTrue(np.issubdtype(preds.dtype, np.integer))

    def test_feature_cols_explicit_wins(self):
        model = cal.CalibratedChurnModel(ProbColumnModel(), feature_cols=["a", "b"])
        self.assertEqual(model.feature_cols, ["a", "b"])

    def test_feature_cols_inherited_from_base_model(self):
        self.assertEqual(cal.CalibratedChurnModel(ProbColumnModel()).feature_cols, ["p"])

    def test_feature_cols_empty_when_unavailable(self):
        class NoCols:
            def predict_proba(self, X):
                return np.zeros(len(X))

        self.assertEqual(cal.CalibratedChurnModel(NoCols()).feature_cols, [])

    def test_save_load_round_trip(self):
        X, y = make_overconfident_data(n=500)
        platt = cal.PlattScaling().fit(X["p"], y)
        original = cal.CalibratedChurnModel(
            ProbColumnModel(), platt, method="sigmoid", threshold=0.37, model_name="LightGBM"
        )
        with tempfile.TemporaryDirectory() as d:
            path = original.save(Path(d) / "nested" / "dir" / "model.joblib")  # parent dirs created
            self.assertTrue(path.exists())
            loaded = cal.CalibratedChurnModel.load(path)

        self.assertEqual(loaded.method, "sigmoid")
        self.assertEqual(loaded.threshold, 0.37)
        self.assertEqual(loaded.model_name, "LightGBM")
        self.assertEqual(loaded.feature_cols, ["p"])
        np.testing.assert_allclose(loaded.predict_proba(X), original.predict_proba(X))
        np.testing.assert_array_equal(loaded.predict(X), original.predict(X))

    def test_saved_artifact_contains_metadata(self):
        model = cal.CalibratedChurnModel(ProbColumnModel(), model_name="LightGBM")
        with tempfile.TemporaryDirectory() as d:
            bundle = joblib.load(model.save(Path(d) / "m.joblib"))
        self.assertEqual(bundle["target_value"], 1)
        self.assertEqual(bundle["probability_definition"], "P(churn_label == 1)")
        self.assertEqual(bundle["source_positive_class"], 1)
        for key in ("base_model", "calibrator", "method", "threshold", "model_name", "feature_cols"):
            self.assertIn(key, bundle)

    def test_load_applies_defaults_for_missing_optional_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "minimal.joblib"
            joblib.dump({"base_model": ProbColumnModel()}, path)
            loaded = cal.CalibratedChurnModel.load(path)
        self.assertIsNone(loaded.calibrator)
        self.assertEqual(loaded.method, "none")
        self.assertEqual(loaded.threshold, 0.5)
        self.assertEqual(loaded.model_name, "unknown")

    def test_load_missing_base_model_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.joblib"
            joblib.dump({"method": "none"}, path)
            with self.assertRaises(KeyError):
                cal.CalibratedChurnModel.load(path)


# ---------------------------------------------------------------------------
# load_recommended_tune_metric
# ---------------------------------------------------------------------------

class TestLoadRecommendedTuneMetric(QuietTestCase):
    def _write_comparison(self, rows):
        pd.DataFrame(rows).to_csv(self.tmp / "best_threshold_comparison_val.csv", index=False)

    def _write_lgbm_only(self, rows):
        pd.DataFrame(rows).to_csv(self.tmp / "best_thresholds_LightGBM_val.csv", index=False)

    def test_no_files_returns_default_and_warns(self):
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp, "recall"), "recall")
        self.assertTrue(self.logged("WARNING"))

    def test_comparison_picks_metric_with_widest_lightgbm_lead(self):
        self._write_comparison([
            {"metric": "f1", "better": "LightGBM", "score_gap": 0.02},
            {"metric": "recall", "better": "LightGBM", "score_gap": 0.08},
            {"metric": "precision", "better": "LogisticRegression", "score_gap": 0.30},
        ])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp), "recall")

    def test_comparison_takes_priority_over_lightgbm_only_file(self):
        self._write_comparison([{"metric": "f2", "better": "LightGBM", "score_gap": 0.05}])
        self._write_lgbm_only([{"metric": "f1", "best_score": 0.99}])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp), "f2")

    def test_unsupported_metric_in_comparison_falls_back_to_lightgbm_only(self):
        self._write_comparison([{"metric": "mcc", "better": "LightGBM", "score_gap": 0.5}])
        self._write_lgbm_only([{"metric": "f1", "best_score": 0.7}])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp), "f1")
        self.assertTrue(self.logged("WARNING"))

    def test_lightgbm_never_leads_falls_back_to_lightgbm_only(self):
        self._write_comparison([{"metric": "f1", "better": "LogisticRegression", "score_gap": 0.1}])
        self._write_lgbm_only([
            {"metric": "f1", "best_score": 0.60},
            {"metric": "balanced_accuracy", "best_score": 0.82},
        ])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp), "balanced_accuracy")

    def test_lightgbm_never_leads_and_no_other_file_returns_default(self):
        self._write_comparison([{"metric": "f1", "better": "LogisticRegression", "score_gap": 0.1}])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp, "accuracy"), "accuracy")

    def test_lightgbm_only_picks_highest_best_score(self):
        self._write_lgbm_only([
            {"metric": "f1", "best_score": 0.55},
            {"metric": "precision", "best_score": 0.91},
            {"metric": "recall", "best_score": 0.70},
        ])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp), "precision")

    def test_unsupported_metric_in_lightgbm_only_returns_default(self):
        self._write_lgbm_only([{"metric": "mcc", "best_score": 0.9}])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp, "f1"), "f1")

    def test_malformed_comparison_csv_is_survivable(self):
        # Missing the 'better'/'score_gap' columns -> KeyError internally, must not propagate.
        self._write_comparison([{"metric": "f1", "unrelated": 1}])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp, "recall"), "recall")

    def test_malformed_comparison_csv_still_allows_lightgbm_only_fallback(self):
        self._write_comparison([{"oops": 1}])
        self._write_lgbm_only([{"metric": "f2", "best_score": 0.6}])
        self.assertEqual(cal.load_recommended_tune_metric(self.tmp), "f2")

    def test_returned_metric_is_always_supported(self):
        self._write_comparison([{"metric": "recall", "better": "LightGBM", "score_gap": 0.1}])
        self.assertIn(cal.load_recommended_tune_metric(self.tmp), cal.SUPPORTED_TUNE_METRICS)


# ---------------------------------------------------------------------------
# compare_calibration_methods / calibrate_model
# ---------------------------------------------------------------------------

class TestCompareCalibrationMethods(QuietTestCase):
    def setUp(self):
        super().setUp()
        self.X, self.y = make_overconfident_data()

    def test_returns_table_with_all_three_methods(self):
        df, calibrators = cal.compare_calibration_methods(ProbColumnModel(), self.X, self.y)
        self.assertEqual(set(df["method"]), {"none", "sigmoid", "isotonic"})
        self.assertEqual(set(calibrators), {"none", "sigmoid", "isotonic"})
        self.assertEqual(
            list(df.columns), ["method", "brier_score", "log_loss", "ece", "mce"]
        )

    def test_table_sorted_by_brier_ascending(self):
        df, _ = cal.compare_calibration_methods(ProbColumnModel(), self.X, self.y)
        self.assertTrue(df["brier_score"].is_monotonic_increasing)

    def test_calibrator_types(self):
        _, calibrators = cal.compare_calibration_methods(ProbColumnModel(), self.X, self.y)
        self.assertIsNone(calibrators["none"])
        self.assertIsInstance(calibrators["sigmoid"], cal.PlattScaling)
        self.assertIsInstance(calibrators["isotonic"], cal.IsotonicCalibration)
        self.assertTrue(calibrators["sigmoid"].is_fitted_)
        self.assertTrue(calibrators["isotonic"].is_fitted_)

    def test_calibration_beats_raw_on_overconfident_model(self):
        df, _ = cal.compare_calibration_methods(ProbColumnModel(), self.X, self.y)
        brier = df.set_index("method")["brier_score"]
        self.assertLess(brier["sigmoid"], brier["none"])
        self.assertLess(brier["isotonic"], brier["none"])

    def test_inputs_are_not_mutated(self):
        X_before, y_before = self.X.copy(), self.y.copy()
        cal.compare_calibration_methods(ProbColumnModel(), self.X, self.y)
        pd.testing.assert_frame_equal(self.X, X_before)
        pd.testing.assert_series_equal(self.y, y_before)


class TestCalibrateModel(QuietTestCase):
    def setUp(self):
        super().setUp()
        self.X, self.y = make_overconfident_data()
        patcher = mock.patch.object(cal, "find_optimal_threshold", return_value=(0.42, 0.77))
        self.find_threshold = patcher.start()
        self.addCleanup(patcher.stop)

    def test_selects_a_real_calibrator_for_overconfident_model(self):
        model, comparison = cal.calibrate_model(ProbColumnModel(), self.X, self.y, "LightGBM")
        self.assertIn(model.method, {"sigmoid", "isotonic"})
        self.assertIsNotNone(model.calibrator)
        self.assertEqual(model.method, comparison.iloc[0]["method"])

    def test_calibration_reduces_validation_ece(self):
        model, _ = cal.calibrate_model(ProbColumnModel(), self.X, self.y, "LightGBM")
        before = cal.calibration_metrics(self.y, model.raw_predict_proba(self.X))
        after = cal.calibration_metrics(self.y, model.predict_proba(self.X))
        self.assertLess(after["ece"], before["ece"])
        self.assertLess(after["brier_score"], before["brier_score"])

    def test_threshold_comes_from_retuning_on_calibrated_probs(self):
        model, _ = cal.calibrate_model(
            ProbColumnModel(), self.X, self.y, "LightGBM", tune_metric="recall"
        )
        self.assertEqual(model.threshold, 0.42)

        self.find_threshold.assert_called_once()
        args, kwargs = self.find_threshold.call_args
        np.testing.assert_array_equal(args[0], self.y.to_numpy())
        # The probabilities handed to the tuner must be the CALIBRATED ones,
        # not the raw scores -- that is the whole point of re-tuning.
        np.testing.assert_allclose(args[1], model.predict_proba(self.X))
        self.assertFalse(np.allclose(args[1], self.X["p"].to_numpy()))
        self.assertEqual(kwargs["metric"], "recall")

    def test_default_tune_metric_is_balanced_accuracy(self):
        cal.calibrate_model(ProbColumnModel(), self.X, self.y)
        self.assertEqual(self.find_threshold.call_args.kwargs["metric"], "balanced_accuracy")

    def test_model_name_is_propagated(self):
        model, _ = cal.calibrate_model(ProbColumnModel(), self.X, self.y, model_name="MyModel")
        self.assertEqual(model.model_name, "MyModel")

    def test_calibration_cannot_help_when_scores_already_match_base_rate(self):
        # Constant score equal to the base rate: nothing to improve.
        y = pd.Series([1] * 30 + [0] * 70)
        X = pd.DataFrame({"p": [0.3] * 100})
        model, _ = cal.calibrate_model(ProbColumnModel(), X, y, "LightGBM")
        self.assertEqual(model.method, "none")
        self.assertIsNone(model.calibrator)

    def _fake_comparison(self, best_brier, none_brier):
        """Force a comparison table where 'sigmoid' ranks first."""
        df = pd.DataFrame(
            [
                {"method": "sigmoid", "brier_score": best_brier, "log_loss": 0.5, "ece": 0.01, "mce": 0.02},
                {"method": "none", "brier_score": none_brier, "log_loss": 0.6, "ece": 0.05, "mce": 0.09},
            ]
        )
        calibrators = {"none": None, "sigmoid": HalvingCalibrator(), "isotonic": None}
        return mock.patch.object(cal, "compare_calibration_methods", return_value=(df, calibrators))

    def test_negligible_improvement_reverts_to_uncalibrated(self):
        # Improvement of 5e-5 is below the 1e-4 cut-off -> keep raw probabilities.
        with self._fake_comparison(best_brier=0.20995, none_brier=0.21):
            model, _ = cal.calibrate_model(ProbColumnModel(), self.X, self.y, "LightGBM")
        self.assertEqual(model.method, "none")
        self.assertIsNone(model.calibrator)
        self.assertTrue(any("negligible" in m for m in self.logged("WARNING")))
        # With no calibrator, the tuner sees the raw probabilities untouched.
        np.testing.assert_allclose(self.find_threshold.call_args.args[1], self.X["p"].to_numpy())

    def test_meaningful_improvement_keeps_calibrator(self):
        # Improvement of 1e-3 is above the cut-off -> keep the calibrator.
        with self._fake_comparison(best_brier=0.209, none_brier=0.21):
            model, _ = cal.calibrate_model(ProbColumnModel(), self.X, self.y, "LightGBM")
        self.assertEqual(model.method, "sigmoid")
        self.assertIsInstance(model.calibrator, HalvingCalibrator)
        self.assertFalse(self.logged("WARNING"))
        np.testing.assert_allclose(
            self.find_threshold.call_args.args[1], self.X["p"].to_numpy() * 0.5
        )


# ---------------------------------------------------------------------------
# plot_reliability_diagram / evaluate_on_test
# ---------------------------------------------------------------------------

class TestPlotReliabilityDiagram(QuietTestCase):
    def test_writes_png_file(self):
        X, y = make_overconfident_data(n=500)
        out = self.tmp / "plot.png"
        cal.plot_reliability_diagram(
            {"raw": (y.to_numpy(), X["p"].to_numpy())}, "title", out, n_bins=5
        )
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_multiple_curves_supported(self):
        X, y = make_overconfident_data(n=500)
        out = self.tmp / "multi.png"
        curves = {
            "a": (y.to_numpy(), X["p"].to_numpy()),
            "b": (y.to_numpy(), np.clip(X["p"].to_numpy() * 0.9, 0, 1)),
        }
        cal.plot_reliability_diagram(curves, "multi", str(out))  # str path accepted
        self.assertTrue(out.exists())


class TestEvaluateOnTest(QuietTestCase):
    def setUp(self):
        super().setUp()
        _, self.report_dir = self.use_tmp_output_dirs()
        self.X, self.y = make_overconfident_data(seed=7)

        patcher = mock.patch.object(cal, "calculate_metrics", return_value={"f1": 0.5})
        self.calc_metrics = patcher.start()
        self.addCleanup(patcher.stop)

        patcher = mock.patch.object(cal, "format_metrics_summary", return_value="SUMMARY")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _model(self, with_calibrator=True, threshold=0.5):
        calibrator = None
        method = "none"
        if with_calibrator:
            Xv, yv = make_overconfident_data(seed=99)
            calibrator = cal.PlattScaling().fit(Xv["p"], yv)
            method = "sigmoid"
        return cal.CalibratedChurnModel(
            ProbColumnModel(), calibrator, method=method, threshold=threshold, model_name="LightGBM"
        )

    def test_result_structure(self):
        result = cal.evaluate_on_test(self._model(), self.X, self.y)
        self.assertEqual(
            set(result),
            {
                "classification_metrics", "calibration_before", "calibration_after",
                "threshold", "method", "reliability_plot",
            },
        )
        self.assertEqual(result["method"], "sigmoid")
        self.assertEqual(result["classification_metrics"], {"f1": 0.5})

    def test_classification_metrics_computed_on_calibrated_predictions(self):
        model = self._model(threshold=0.4)
        cal.evaluate_on_test(model, self.X, self.y)
        kwargs = self.calc_metrics.call_args.kwargs
        np.testing.assert_array_equal(kwargs["y_true"], self.y.to_numpy())
        np.testing.assert_array_equal(kwargs["y_pred"], model.predict(self.X))
        np.testing.assert_allclose(kwargs["y_prob"], model.predict_proba(self.X))
        self.assertEqual(kwargs["threshold"], 0.4)

    def test_calibration_improves_on_test_set(self):
        result = cal.evaluate_on_test(self._model(), self.X, self.y)
        self.assertLess(result["calibration_after"]["ece"], result["calibration_before"]["ece"])
        self.assertLess(
            result["calibration_after"]["brier_score"], result["calibration_before"]["brier_score"]
        )

    def test_without_calibrator_before_equals_after(self):
        result = cal.evaluate_on_test(self._model(with_calibrator=False), self.X, self.y)
        self.assertEqual(result["calibration_before"], result["calibration_after"])

    def test_reliability_plot_is_written_to_report_dir(self):
        result = cal.evaluate_on_test(self._model(), self.X, self.y)
        plot = Path(result["reliability_plot"])
        self.assertTrue(plot.exists())
        self.assertEqual(plot.parent, self.report_dir)
        self.assertIn("LightGBM", plot.name)
        self.assertIn("20240101_000000", plot.name)

    def test_plot_gets_uncalibrated_and_calibrated_curves(self):
        with mock.patch.object(cal, "plot_reliability_diagram") as plot:
            cal.evaluate_on_test(self._model(), self.X, self.y, n_bins=5)
        curves = plot.call_args.kwargs["curves"]
        self.assertEqual(list(curves), ["Uncalibrated", "Calibrated (sigmoid)"])
        self.assertEqual(plot.call_args.kwargs["n_bins"], 5)


# ---------------------------------------------------------------------------
# _LGBAdapter
# ---------------------------------------------------------------------------

class TestLGBAdapter(QuietTestCase):
    def setUp(self):
        super().setUp()
        self.sk = StubSkModel()
        self.X = pd.DataFrame({"p": [0.1, 0.7, 0.9]})

    def test_default_positive_class_is_churn_and_not_inverted(self):
        adapter = cal._LGBAdapter(self.sk, ["p"])
        self.assertEqual(adapter.positive_class, 1)
        self.assertFalse(adapter.invert_proba)
        np.testing.assert_allclose(adapter.predict_proba(self.X), [0.1, 0.7, 0.9])

    def test_positive_class_one_is_not_inverted(self):
        adapter = cal._LGBAdapter(self.sk, ["p"], positive_class=1)
        self.assertFalse(adapter.invert_proba)
        self.print_mock.assert_not_called()

    def test_positive_class_zero_inverts_probabilities(self):
        adapter = cal._LGBAdapter(self.sk, ["p"], positive_class=0)
        self.assertTrue(adapter.invert_proba)
        np.testing.assert_allclose(adapter.predict_proba(self.X), [0.9, 0.3, 0.1])

    def test_inversion_prints_a_note(self):
        cal._LGBAdapter(self.sk, ["p"], positive_class=0)
        self.print_mock.assert_called_once()
        self.assertIn("positive_class=0", self.print_mock.call_args.args[0])

    def test_dataframe_input_selects_and_orders_feature_cols(self):
        df = pd.DataFrame({"extra": [9, 9], "b": [0.4, 0.6], "a": [0.1, 0.2]})
        cal._LGBAdapter(self.sk, ["a", "b"]).predict_proba(df)
        np.testing.assert_allclose(self.sk.last_X, [[0.1, 0.4], [0.2, 0.6]])

    def test_dataframe_missing_feature_column_raises(self):
        with self.assertRaises(KeyError):
            cal._LGBAdapter(self.sk, ["missing"]).predict_proba(self.X)

    def test_numpy_input_passes_through_unchanged(self):
        arr = np.array([[0.2], [0.8]])
        out = cal._LGBAdapter(self.sk, ["p"]).predict_proba(arr)
        np.testing.assert_allclose(self.sk.last_X, arr)
        np.testing.assert_allclose(out, [0.2, 0.8])

    def test_output_is_1d(self):
        self.assertEqual(cal._LGBAdapter(self.sk, ["p"]).predict_proba(self.X).shape, (3,))


# ---------------------------------------------------------------------------
# load_latest_lightgbm_model
# ---------------------------------------------------------------------------

class TestLoadLatestLightGBMModel(QuietTestCase):
    def setUp(self):
        super().setUp()
        self.model_dir, _ = self.use_tmp_output_dirs()

    def _save(self, name, bundle):
        joblib.dump(bundle, self.model_dir / name)

    @staticmethod
    def _bundle(positive_class=1, **extra):
        return {"model": StubSkModel(), "feature_cols": ["p"], "positive_class": positive_class, **extra}

    def test_no_artifacts_returns_none_with_warning(self):
        self.assertIsNone(cal.load_latest_lightgbm_model())
        self.assertTrue(self.logged("WARNING"))

    def test_loads_valid_artifact_into_adapter(self):
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=1))
        adapter = cal.load_latest_lightgbm_model()
        self.assertIsInstance(adapter, cal._LGBAdapter)
        self.assertEqual(adapter.feature_cols, ["p"])
        self.assertEqual(adapter.positive_class, 1)

    def test_picks_newest_artifact_by_timestamp(self):
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=1))
        self._save("lgb_churn_model_20240301_120000.joblib", self._bundle(positive_class=0))
        self._save("lgb_churn_model_20240201_000000.joblib", self._bundle(positive_class=1))
        self.assertEqual(cal.load_latest_lightgbm_model().positive_class, 0)

    def test_ignores_calibrated_artifacts(self):
        self._save("lgb_churn_model_calibrated.joblib", self._bundle(positive_class=0))
        self.assertIsNone(cal.load_latest_lightgbm_model())

    def test_calibrated_file_does_not_shadow_a_real_training_artifact(self):
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=1))
        self._save("lgb_churn_model_calibrated.joblib", self._bundle(positive_class=0))
        self.assertEqual(cal.load_latest_lightgbm_model().positive_class, 1)

    def test_ignores_unrelated_files(self):
        self._save("some_other_model.joblib", self._bundle())
        self.assertIsNone(cal.load_latest_lightgbm_model())

    def test_corrupt_newest_file_is_skipped_for_older_valid_one(self):
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=1))
        (self.model_dir / "lgb_churn_model_20240201_000000.joblib").write_bytes(b"not a joblib file")
        adapter = cal.load_latest_lightgbm_model()
        self.assertIsNotNone(adapter)
        self.assertTrue(any("corrupt" in m.lower() for m in self.logged("WARNING")))

    def test_wrong_schema_is_skipped(self):
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=1))
        self._save("lgb_churn_model_20240201_000000.joblib", {"unexpected": True})
        self._save("lgb_churn_model_20240301_000000.joblib", ["not", "a", "dict"])
        self.assertIsNotNone(cal.load_latest_lightgbm_model())
        self.assertGreaterEqual(
            sum("unexpected artifact schema" in m for m in self.logged("WARNING")), 2
        )

    def test_missing_model_or_feature_cols_key_is_wrong_schema(self):
        self._save("lgb_churn_model_20240101_000000.joblib", {"model": StubSkModel(), "positive_class": 1})
        self.assertIsNone(cal.load_latest_lightgbm_model())

    def test_only_invalid_files_returns_none(self):
        (self.model_dir / "lgb_churn_model_20240101_000000.joblib").write_bytes(b"garbage")
        self.assertIsNone(cal.load_latest_lightgbm_model())
        self.assertTrue(any("No valid" in m for m in self.logged("WARNING")))

    def test_missing_positive_class_raises(self):
        bundle = self._bundle()
        del bundle["positive_class"]
        self._save("lgb_churn_model_20240101_000000.joblib", bundle)
        with self.assertRaisesRegex(ValueError, "positive_class"):
            cal.load_latest_lightgbm_model()

    def test_none_positive_class_raises(self):
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=None))
        with self.assertRaises(ValueError):
            cal.load_latest_lightgbm_model()

    def test_invalid_positive_class_raises(self):
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=2))
        with self.assertRaisesRegex(ValueError, "expected 0 or 1"):
            cal.load_latest_lightgbm_model()

    def test_missing_positive_class_raises_even_if_older_artifact_is_valid(self):
        # Must fail loudly rather than silently fall back to an older file.
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=1))
        newer = self._bundle()
        del newer["positive_class"]
        self._save("lgb_churn_model_20240201_000000.joblib", newer)
        with self.assertRaises(ValueError):
            cal.load_latest_lightgbm_model()

    def test_positive_class_zero_is_accepted(self):
        # 0 is falsy: ensure "is None" (not truthiness) is what's checked.
        self._save("lgb_churn_model_20240101_000000.joblib", self._bundle(positive_class=0))
        adapter = cal.load_latest_lightgbm_model()
        self.assertEqual(adapter.positive_class, 0)
        self.assertTrue(adapter.invert_proba)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain(QuietTestCase):
    def setUp(self):
        super().setUp()
        self.model_dir, self.report_dir = self.use_tmp_output_dirs()

        X_val, y_val = make_overconfident_data(seed=1)
        X_test, y_test = make_overconfident_data(seed=2)
        X_train, y_train = make_overconfident_data(n=200, seed=3)
        self.data = (X_train, y_train, X_val, y_val, X_test, y_test)

        self.patches = {
            "load_model_ready_data": mock.patch.object(cal, "load_model_ready_data", return_value=self.data),
            "load_latest_lightgbm_model": mock.patch.object(
                cal, "load_latest_lightgbm_model", return_value=ProbColumnModel()
            ),
            "find_optimal_threshold": mock.patch.object(
                cal, "find_optimal_threshold", return_value=(0.45, 0.7)
            ),
            "calculate_metrics": mock.patch.object(
                cal, "calculate_metrics", return_value={"f1": 0.6, "roc_auc": 0.8, "pr_auc": 0.7}
            ),
            "format_metrics_summary": mock.patch.object(cal, "format_metrics_summary", return_value="S"),
            "log_experiment": mock.patch.object(cal, "log_experiment"),
        }
        self.mocks = {}
        for name, p in self.patches.items():
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    def test_happy_path_returns_results_and_writes_artifacts(self):
        result = cal.main()

        self.assertEqual(list(result), ["LightGBM"])
        artifact = self.model_dir / "lgb_churn_model_calibrated.joblib"
        self.assertTrue(artifact.exists())
        self.assertTrue((self.report_dir / "calibration_comparison_20240101_000000.csv").exists())

        loaded = cal.CalibratedChurnModel.load(artifact)
        self.assertEqual(loaded.threshold, 0.45)
        self.assertEqual(loaded.model_name, "LightGBM")

    def test_experiment_is_logged_with_final_settings(self):
        cal.main()
        kwargs = self.mocks["log_experiment"].call_args.kwargs
        self.assertEqual(kwargs["model_name"], "LightGBM")
        self.assertEqual(kwargs["threshold"], 0.45)
        self.assertEqual(kwargs["features_count"], self.data[0].shape[1])
        self.assertTrue(kwargs["artifacts_path"].endswith("lgb_churn_model_calibrated.joblib"))
        self.assertEqual(kwargs["metrics"], {"f1": 0.6, "roc_auc": 0.8, "pr_auc": 0.7})

    def test_experiment_logging_failure_does_not_crash_pipeline(self):
        self.mocks["log_experiment"].side_effect = RuntimeError("db down")
        result = cal.main()
        self.assertIn("LightGBM", result)
        self.assertTrue(any("Could not log" in m for m in self.logged("WARNING")))

    def test_tune_metric_from_threshold_analysis_reaches_threshold_search(self):
        pd.DataFrame([{"metric": "f1", "best_score": 0.8}]).to_csv(
            self.report_dir / "best_thresholds_LightGBM_val.csv", index=False
        )
        cal.main()
        self.assertEqual(self.mocks["find_optimal_threshold"].call_args.kwargs["metric"], "f1")

    def test_default_tune_metric_used_when_no_analysis_exists(self):
        cal.main()
        self.assertEqual(
            self.mocks["find_optimal_threshold"].call_args.kwargs["metric"], "balanced_accuracy"
        )

    def test_exits_when_data_cannot_be_loaded(self):
        self.mocks["load_model_ready_data"].side_effect = RuntimeError("no db")
        with self.assertRaises(SystemExit) as ctx:
            cal.main()
        self.assertEqual(ctx.exception.code, 1)
        self.mocks["load_latest_lightgbm_model"].assert_not_called()

    def test_exits_when_no_model_available(self):
        self.mocks["load_latest_lightgbm_model"].return_value = None
        with self.assertRaises(SystemExit) as ctx:
            cal.main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertFalse((self.model_dir / "lgb_churn_model_calibrated.joblib").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
    
