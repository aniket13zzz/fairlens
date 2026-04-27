"""
FairLens AI — Bias Detection Engine
Metrics: Disparate Impact, Demographic Parity, Statistical Parity,
         Equal Opportunity (TPR gap), Predictive Parity (PPV gap).
Proxy feature detection: Pearson (numeric) + Cramér's V (categorical).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .errors import ColumnNotFound, InvalidDataset


class BiasDetector:
    BIAS_THRESHOLDS = {
        "disparate_impact": {"severe": 0.6, "moderate": 0.75, "mild": 0.9},
        "demographic_parity_diff": {"severe": 0.2, "moderate": 0.1, "mild": 0.05},
    }

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_col: str,
        positive_outcome: str,
        prediction_col: Optional[str] = None,
        ground_truth_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full bias analysis. Returns comprehensive metrics dict."""

        available = df.columns.tolist()
        if target_col not in df.columns:
            raise ColumnNotFound(target_col, available)
        if sensitive_col not in df.columns:
            raise ColumnNotFound(sensitive_col, available)

        df = df.copy()
        df[target_col] = df[target_col].astype(str)
        df[sensitive_col] = df[sensitive_col].astype(str)
        positive_outcome = str(positive_outcome)

        if positive_outcome not in df[target_col].unique():
            raise InvalidDataset(
                f"Positive outcome '{positive_outcome}' not found in column '{target_col}'. "
                f"Values found: {df[target_col].unique().tolist()[:10]}"
            )

        groups = df[sensitive_col].unique().tolist()
        group_stats = self._compute_group_stats(df, target_col, sensitive_col, positive_outcome)
        metrics = self._compute_bias_metrics(group_stats, groups)

        # Optional: Equal Opportunity + Predictive Parity
        advanced_metrics = {}
        if prediction_col and ground_truth_col:
            advanced_metrics = self._compute_advanced_metrics(
                df, prediction_col, ground_truth_col, sensitive_col, positive_outcome
            )

        severity, severity_score = self._assess_severity(metrics)
        affected = self._estimate_affected(group_stats, metrics)
        feature_importance = self._compute_proxy_features(df, target_col, sensitive_col)

        return {
            "status": "analysis_complete",
            "dataset_info": {
                "total_rows": len(df),
                "groups": groups,
                "sensitive_attribute": sensitive_col,
                "target_attribute": target_col,
                "positive_label": positive_outcome,
            },
            "group_stats": group_stats,
            "metrics": {**metrics, **advanced_metrics},
            "severity": severity,
            "severity_score": severity_score,
            "is_biased": severity in ("SEVERE", "MODERATE"),
            "affected_estimate": affected,
            "feature_importance": feature_importance,
        }

    # ── private helpers ─────────────────────────────────────────────────────

    def _compute_group_stats(
        self,
        df: pd.DataFrame,
        target_col: str,
        sensitive_col: str,
        positive_outcome: str,
    ) -> Dict[str, Dict]:
        stats = {}
        for group in df[sensitive_col].unique():
            gdf = df[df[sensitive_col] == group]
            n = len(gdf)
            selected = int((gdf[target_col] == positive_outcome).sum())
            rate = selected / n if n > 0 else 0.0
            stats[str(group)] = {
                "count": n,
                "selected": selected,
                "selection_rate": round(float(rate), 4),
                "percentage": round(float(rate * 100), 2),
                "small_sample_warning": n < 30,
            }
        return stats

    def _compute_bias_metrics(
        self, group_stats: Dict, groups: List[str]
    ) -> Dict[str, Any]:
        rates = {g: group_stats[str(g)]["selection_rate"] for g in groups}
        max_rate = max(rates.values())
        min_rate = min(rates.values())

        privileged = max(rates, key=rates.get)
        unprivileged = min(rates, key=rates.get)
        mean_rate = float(np.mean(list(rates.values())))

        disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0
        dp_diff = max_rate - min_rate
        statistical_parity = {g: round(rates[g] - mean_rate, 4) for g in rates}

        return {
            "disparate_impact": round(float(disparate_impact), 4),
            "demographic_parity_diff": round(float(dp_diff), 4),
            "privileged_group": str(privileged),
            "unprivileged_group": str(unprivileged),
            "max_selection_rate": round(float(max_rate), 4),
            "min_selection_rate": round(float(min_rate), 4),
            "mean_selection_rate": round(mean_rate, 4),
            "statistical_parity": statistical_parity,
            "passes_80_rule": disparate_impact >= 0.8,
            "four_fifths_threshold": 0.8,
            "selection_rate_ratio": round(float(disparate_impact * 100), 1),
        }

    def _compute_advanced_metrics(
        self,
        df: pd.DataFrame,
        prediction_col: str,
        ground_truth_col: str,
        sensitive_col: str,
        positive_outcome: str,
    ) -> Dict[str, Any]:
        """Equal Opportunity (TPR gap) + Predictive Parity (PPV gap)."""
        results: Dict[str, Any] = {}
        tpr_by_group: Dict[str, float] = {}
        ppv_by_group: Dict[str, float] = {}

        for group in df[sensitive_col].unique():
            gdf = df[df[sensitive_col] == group]
            y_true = (gdf[ground_truth_col].astype(str) == positive_outcome)
            y_pred = (gdf[prediction_col].astype(str) == positive_outcome)

            tp = (y_true & y_pred).sum()
            fp = (~y_true & y_pred).sum()
            fn = (y_true & ~y_pred).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            tpr_by_group[str(group)] = round(float(tpr), 4)
            ppv_by_group[str(group)] = round(float(ppv), 4)

        if tpr_by_group:
            tpr_vals = list(tpr_by_group.values())
            ppv_vals = list(ppv_by_group.values())
            results["equal_opportunity_diff"] = round(max(tpr_vals) - min(tpr_vals), 4)
            results["predictive_parity_diff"] = round(max(ppv_vals) - min(ppv_vals), 4)
            results["tpr_by_group"] = tpr_by_group
            results["ppv_by_group"] = ppv_by_group
            results["passes_equal_opportunity"] = results["equal_opportunity_diff"] < 0.1
            results["passes_predictive_parity"] = results["predictive_parity_diff"] < 0.1

        return results

    def _assess_severity(self, metrics: Dict) -> Tuple[str, int]:
        di = metrics["disparate_impact"]
        dp = metrics["demographic_parity_diff"]
        score = 0

        if di < 0.6:
            score += 40
        elif di < 0.75:
            score += 25
        elif di < 0.8:
            score += 15
        elif di < 0.9:
            score += 8

        if dp > 0.3:
            score += 40
        elif dp > 0.2:
            score += 30
        elif dp > 0.1:
            score += 20
        elif dp > 0.05:
            score += 10

        if not metrics["passes_80_rule"]:
            score += 20

        score = min(100, score)

        if score >= 60:
            return "SEVERE", score
        elif score >= 30:
            return "MODERATE", score
        elif score >= 10:
            return "MILD", score
        return "FAIR", score

    def _estimate_affected(
        self, group_stats: Dict, metrics: Dict
    ) -> Dict[str, Any]:
        rates = {g: group_stats[g]["selection_rate"] for g in group_stats}
        max_rate = max(rates.values())
        unprivileged = metrics["unprivileged_group"]
        unpriv_rate = rates[unprivileged]
        unpriv_count = group_stats[unprivileged]["count"]
        gap = max_rate - unpriv_rate
        unfairly_rejected = int(round(gap * unpriv_count))
        return {
            "group": unprivileged,
            "unfairly_impacted": unfairly_rejected,
            "selection_gap_pct": round(gap * 100, 1),
            "narrative": (
                f"If deployed, this system would unfairly disadvantage "
                f"{unfairly_rejected} qualified {unprivileged} candidates "
                f"in this dataset alone."
            ),
        }

    def _compute_proxy_features(
        self, df: pd.DataFrame, target_col: str, sensitive_col: str
    ) -> List[Dict]:
        """Detect proxy features for sensitive attribute.
        Numeric columns → Pearson |r|.
        Categorical columns → Cramér's V (chi² based).
        """
        results = []
        sensitive = df[sensitive_col].astype(str)

        for col in df.columns:
            if col in (target_col, sensitive_col):
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    sensitive_encoded = pd.Categorical(sensitive).codes
                    corr = abs(float(np.corrcoef(df[col].fillna(0), sensitive_encoded)[0, 1]))
                    method = "pearson"
                else:
                    corr = self._cramers_v(df[col].astype(str), sensitive)
                    method = "cramers_v"

                results.append({
                    "feature": col,
                    "correlation_with_sensitive": round(corr, 3),
                    "method": method,
                    "risk_level": "HIGH" if corr > 0.5 else "MEDIUM" if corr > 0.3 else "LOW",
                })
            except Exception:
                pass

        results.sort(key=lambda x: x["correlation_with_sensitive"], reverse=True)
        return results[:10]

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Cramér's V — strength of association between two categorical variables."""
        ct = pd.crosstab(x, y)
        if ct.empty:
            return 0.0
        chi2, _, _, _ = chi2_contingency(ct, correction=False)
        n = len(x)
        r, k = ct.shape
        denom = n * (min(r, k) - 1)
        if denom <= 0:
            return 0.0
        return float(np.sqrt(chi2 / denom))
