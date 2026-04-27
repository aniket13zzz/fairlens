"""
FairLens AI — AI Explanation Engine
Generates plain-English bias explanations via Claude, OpenAI, or rule-based fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class AIExplainer:
    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    async def explain(self, analysis: Dict[str, Any]) -> str:
        try:
            if self.anthropic_key:
                return await self._call_anthropic(_bias_prompt(analysis))
            if self.openai_key:
                return await self._call_openai(_bias_prompt(analysis))
        except Exception as exc:
            logger.warning("LLM explanation failed: %s", exc)
        return _fallback_explanation(analysis)

    async def explain_fix(self, original: Dict, fixed: Dict) -> str:
        try:
            if self.anthropic_key:
                return await self._call_anthropic(_fix_prompt(original, fixed))
            if self.openai_key:
                return await self._call_openai(_fix_prompt(original, fixed))
        except Exception as exc:
            logger.warning("LLM fix explanation failed: %s", exc)
        return _fallback_fix_explanation(original, fixed)

    async def _call_anthropic(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 350,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()

    async def _call_openai(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 350,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()


# ── Prompt builders ────────────────────────────────────────────────────────

def _bias_prompt(analysis: Dict) -> str:
    m = analysis.get("metrics", {})
    severity = analysis.get("severity", "UNKNOWN")
    affected = analysis.get("affected_estimate", {})
    return (
        "You are a fairness expert explaining AI bias to a business executive.\n\n"
        f"Severity: {severity}\n"
        f"Disparate Impact Ratio: {m.get('disparate_impact', 'N/A')} (1.0=fair, <0.8=EEOC violation)\n"
        f"Demographic Parity Difference: {m.get('demographic_parity_diff', 'N/A')} (0=fair)\n"
        f"Privileged Group: {m.get('privileged_group', 'N/A')} "
        f"({m.get('max_selection_rate', 0)*100:.1f}% selection rate)\n"
        f"Unprivileged Group: {m.get('unprivileged_group', 'N/A')} "
        f"({m.get('min_selection_rate', 0)*100:.1f}% selection rate)\n"
        f"EEOC 80% Rule: {'PASS' if m.get('passes_80_rule') else 'FAIL'}\n"
        f"Impact: {affected.get('narrative', '')}\n\n"
        "Write 2-3 sentences: (1) what the bias means in plain English, "
        "(2) real-world consequence, (3) legal/ethical risk. "
        "Be direct, use numbers, no jargon. Sound urgent if SEVERE."
    )


def _fix_prompt(original: Dict, fixed: Dict) -> str:
    improvement = fixed.get("improvement", {})
    orig_m = original.get("metrics", {})
    fixed_m = fixed.get("fixed_metrics", {})
    return (
        "Explain in 2-3 plain English sentences what the debiasing algorithm accomplished.\n\n"
        f"Before: DI={orig_m.get('disparate_impact', 0):.2f}, "
        f"DP Diff={orig_m.get('demographic_parity_diff', 0):.2f}\n"
        f"After: DI={fixed_m.get('disparate_impact', 0):.2f}, "
        f"DP Diff={fixed_m.get('demographic_parity_diff', 0):.2f}\n"
        f"Bias reduction: {improvement.get('overall_bias_reduction_pct', 0):.1f}%\n"
        f"EEOC compliant: {improvement.get('eeoc_compliant', False)}\n"
        f"Algorithm used: {fixed.get('algorithm', 'Reweighing')}\n\n"
        "Be specific and factual. Explain business and legal benefit."
    )


# ── Rule-based fallbacks ───────────────────────────────────────────────────

def _fallback_explanation(analysis: Dict) -> str:
    m = analysis.get("metrics", {})
    severity = analysis.get("severity", "UNKNOWN")
    di = m.get("disparate_impact", 1.0)
    dp = m.get("demographic_parity_diff", 0.0)
    priv = m.get("privileged_group", "Group A")
    unpriv = m.get("unprivileged_group", "Group B")
    max_r = m.get("max_selection_rate", 0)
    min_r = m.get("min_selection_rate", 0)
    affected = analysis.get("affected_estimate", {})

    if severity == "SEVERE":
        return (
            f"⚠️ CRITICAL BIAS DETECTED: {priv} members are selected at {max_r*100:.1f}% "
            f"vs only {min_r*100:.1f}% for {unpriv} members — a {dp*100:.1f} percentage point gap. "
            f"The Disparate Impact Ratio of {di:.2f} is well below the EEOC 80% threshold, "
            f"meaning this system is likely illegal under US employment law. "
            f"{affected.get('narrative', 'Immediate remediation required before deployment.')}"
        )
    if severity == "MODERATE":
        return (
            f"⚠️ MODERATE BIAS: A {dp*100:.1f}% selection rate gap exists between "
            f"{priv} ({max_r*100:.1f}%) and {unpriv} ({min_r*100:.1f}%) groups. "
            f"Disparate Impact Ratio of {di:.2f} — "
            f"{'fails' if di < 0.8 else 'barely passes'} the EEOC 80% rule. "
            f"Remediation strongly recommended."
        )
    if severity == "MILD":
        return (
            f"ℹ️ MILD BIAS: A small {dp*100:.1f}% selection rate difference exists between groups. "
            f"Disparate Impact Ratio of {di:.2f} meets legal thresholds but monitoring advised. "
            f"Consider investigating whether non-sensitive features act as proxies."
        )
    return (
        f"✅ FAIR SYSTEM: Selection rates are equitable across all groups (gap: {dp*100:.1f}%). "
        f"Disparate Impact Ratio of {di:.2f} exceeds the EEOC 80% threshold. "
        f"This system appears to make fair decisions with respect to the sensitive attribute."
    )


def _fallback_fix_explanation(original: Dict, fixed: Dict) -> str:
    improvement = fixed.get("improvement", {})
    fixed_m = fixed.get("fixed_metrics", {})
    reduction = improvement.get("overall_bias_reduction_pct", 0)
    new_di = fixed_m.get("disparate_impact", 0)
    eeoc = improvement.get("eeoc_compliant", False)
    algo = fixed.get("algorithm", "Reweighing")
    warning = fixed.get("persistent_bias_warning")

    if warning:
        return f"⚠️ {warning}"

    return (
        f"✅ BIAS REDUCED BY {reduction:.0f}%: The {algo} algorithm adjusted the dataset "
        f"to equalise selection probabilities across groups. "
        f"The new Disparate Impact Ratio of {new_di:.2f} "
        f"{'now meets' if eeoc else 'approaches'} the EEOC 80% legal standard. "
        f"Model accuracy impact is reported separately based on actual cross-validation."
    )
