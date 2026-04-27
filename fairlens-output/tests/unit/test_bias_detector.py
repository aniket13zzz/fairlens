"""
Unit tests for BiasDetector.
Coverage: known-biased, known-fair, edge cases (single group, all-zero rate, NaN).
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.bias_detector import BiasDetector
from backend.errors import ColumnNotFound, InvalidDataset


@pytest.fixture
def biased_df():
    """Known-biased dataset: GroupA selected 100%, GroupB 0%."""
    return pd.DataFrame({
        "outcome": ["1"] * 10 + ["0"] * 10,
        "group": ["A"] * 10 + ["B"] * 10,
    })


@pytest.fixture
def fair_df():
    """Known-fair dataset: both groups selected at exactly equal rates."""
    return pd.DataFrame({
        "outcome": ["1"] * 10 + ["0"] * 10 + ["1"] * 10 + ["0"] * 10,
        "group":   ["A"] * 20 + ["B"] * 20,
    })


@pytest.fixture
def rich_df():
    """Dataset with numeric features for proxy detection."""
    rng = np.random.default_rng(42)
    n = 100
    group = ["A"] * 50 + ["B"] * 50
    outcome = ["1"] * 40 + ["0"] * 10 + ["1"] * 5 + ["0"] * 45
    age = rng.integers(22, 60, n).tolist()
    score = rng.uniform(0, 100, n).tolist()
    return pd.DataFrame({"outcome": outcome, "group": group,
                         "age": age, "score": score})


class TestBiasDetector:
    def test_severe_bias_detected(self, biased_df):
        d = BiasDetector()
        result = d.analyze(biased_df, "outcome", "group", "1")
        assert result["severity"] == "SEVERE"
        assert result["is_biased"] is True
        assert result["metrics"]["disparate_impact"] == 0.0
        assert result["metrics"]["demographic_parity_diff"] == 1.0
        assert result["metrics"]["passes_80_rule"] is False

    def test_fair_system_detected(self, fair_df):
        d = BiasDetector()
        result = d.analyze(fair_df, "outcome", "group", "1")
        assert result["severity"] in ("FAIR", "MILD")
        assert result["metrics"]["disparate_impact"] > 0.7

    def test_column_not_found_raises(self, biased_df):
        d = BiasDetector()
        with pytest.raises(ColumnNotFound):
            d.analyze(biased_df, "nonexistent", "group", "1")

    def test_invalid_positive_outcome_raises(self, biased_df):
        d = BiasDetector()
        with pytest.raises(InvalidDataset):
            d.analyze(biased_df, "outcome", "group", "99")

    def test_group_stats_keys(self, biased_df):
        d = BiasDetector()
        result = d.analyze(biased_df, "outcome", "group", "1")
        assert "A" in result["group_stats"]
        assert "B" in result["group_stats"]
        assert result["group_stats"]["A"]["selection_rate"] == 1.0
        assert result["group_stats"]["B"]["selection_rate"] == 0.0

    def test_no_zero_division_single_group(self):
        """Single group dataset must not crash."""
        df = pd.DataFrame({"outcome": ["1", "0", "1"], "group": ["A", "A", "A"]})
        d = BiasDetector()
        result = d.analyze(df, "outcome", "group", "1")
        assert result["metrics"]["disparate_impact"] == 1.0

    def test_nan_in_features_handled(self, rich_df):
        """NaN in numeric features must not crash proxy detection."""
        rich_df.loc[0, "age"] = float("nan")
        d = BiasDetector()
        result = d.analyze(rich_df, "outcome", "group", "1")
        assert isinstance(result["feature_importance"], list)

    def test_small_sample_warning(self):
        """Groups with < 30 rows get flagged."""
        df = pd.DataFrame({
            "outcome": ["1"] * 20 + ["0"] * 15 + ["1"] * 2,
            "group":   ["A"] * 35 + ["B"] * 2,
        })
        d = BiasDetector()
        result = d.analyze(df, "outcome", "group", "1")
        assert result["group_stats"]["B"]["small_sample_warning"] is True
        assert result["group_stats"]["A"]["small_sample_warning"] is False

    def test_proxy_feature_numeric_detected(self, rich_df):
        d = BiasDetector()
        result = d.analyze(rich_df, "outcome", "group", "1")
        features = [f["feature"] for f in result["feature_importance"]]
        assert "age" in features or "score" in features

    def test_cramers_v_categorical_proxy(self):
        """Categorical proxy feature detected via Cramér's V."""
        df = pd.DataFrame({
            "outcome": ["1"] * 10 + ["0"] * 10,
            "group": ["A"] * 10 + ["B"] * 10,
            "department": ["Eng"] * 10 + ["HR"] * 10,  # perfectly correlated with group
        })
        d = BiasDetector()
        result = d.analyze(df, "outcome", "group", "1")
        dept_feat = next(
            (f for f in result["feature_importance"] if f["feature"] == "department"), None
        )
        assert dept_feat is not None
        assert dept_feat["method"] == "cramers_v"
        assert dept_feat["correlation_with_sensitive"] > 0.5

    def test_affected_estimate_narrative(self, biased_df):
        d = BiasDetector()
        result = d.analyze(biased_df, "outcome", "group", "1")
        narrative = result["affected_estimate"]["narrative"]
        assert "unfairly disadvantage" in narrative
        assert "B" in narrative

    def test_severity_score_range(self, biased_df, fair_df):
        d = BiasDetector()
        biased = d.analyze(biased_df, "outcome", "group", "1")
        fair = d.analyze(fair_df, "outcome", "group", "1")
        assert 0 <= biased["severity_score"] <= 100
        assert biased["severity_score"] >= fair["severity_score"]
