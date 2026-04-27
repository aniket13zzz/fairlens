"""
FairLens AI — PDF Compliance Report Generator
Adds proxy feature risk section (missing from v1).
Temp files are deleted by FastAPI BackgroundTasks — not by this class.
"""
from __future__ import annotations


import tempfile
from datetime import datetime, timezone
from typing import Any, Dict

_RL_AVAILABLE: bool | None = None


def _check_reportlab() -> bool:
    global _RL_AVAILABLE
    if _RL_AVAILABLE is None:
        try:
            import reportlab  # noqa: F401
            _RL_AVAILABLE = True
        except ImportError:
            _RL_AVAILABLE = False
    return _RL_AVAILABLE


class ReportGenerator:
    def generate(
        self,
        session_data: Dict[str, Any],
        org_name: str = "Your Organization",
        auditor_name: str = "FairLens AI System",
    ) -> str:
        """Generate PDF. Returns temp file path. Caller must delete after response."""
        if not _check_reportlab():
            raise ImportError("reportlab not installed. Run: pip install reportlab")

        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        DARK = HexColor("#0a0a1a")
        GREEN = HexColor("#00c853")
        RED = HexColor("#ff1744")
        YELLOW = HexColor("#ffab00")
        LIGHT_GRAY = HexColor("#f5f5f5")
        MED_GRAY = HexColor("#9e9e9e")
        ORANGE = HexColor("#ff6d00")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()

        doc = SimpleDocTemplate(
            tmp.name,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        S = lambda name, **kw: ParagraphStyle(name, parent=styles["Normal"], **kw)  # noqa
        title_s = S("T", fontSize=26, fontName="Helvetica-Bold", textColor=DARK,
                     alignment=TA_CENTER, spaceAfter=6)
        subtitle_s = S("Sub", fontSize=11, textColor=MED_GRAY, alignment=TA_CENTER, spaceAfter=4)
        section_s = S("Sec", fontSize=13, fontName="Helvetica-Bold", textColor=DARK,
                       spaceBefore=14, spaceAfter=8)
        body_s = S("B", fontSize=10, textColor=DARK, spaceAfter=6, leading=16)
        small_s = S("Sm", fontSize=8, textColor=MED_GRAY)
        verdict_s = S("V", fontSize=16, fontName="Helvetica-Bold", textColor=white,
                       alignment=TA_CENTER)
        quote_s = S("Q", fontSize=10, fontName="Helvetica-BoldOblique", alignment=TA_CENTER)

        analysis = session_data.get("analysis", {})
        fixed = session_data.get("fixed")
        metrics = analysis.get("metrics", {})
        severity = analysis.get("severity", "UNKNOWN")
        group_stats = analysis.get("group_stats", {})
        dataset_info = analysis.get("dataset_info", {})
        affected = analysis.get("affected_estimate", {})
        feature_importance = analysis.get("feature_importance", [])

        sev_color = {
            "SEVERE": RED, "MODERATE": YELLOW, "MILD": ORANGE, "FAIR": GREEN
        }.get(severity, MED_GRAY)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        story = []

        # ── Header
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("FairLens AI", title_s))
        story.append(Paragraph("AI Bias Audit Compliance Report", subtitle_s))
        story.append(HRFlowable(width="100%", thickness=2, color=DARK))
        story.append(Spacer(1, 0.3 * cm))

        # ── Meta
        meta = [
            ["Organization:", org_name, "Date:", now],
            ["Auditor:", auditor_name, "Dataset:", session_data.get("filename", "uploaded.csv")],
            ["Sensitive Attr:", dataset_info.get("sensitive_attribute", "—"),
             "Target:", dataset_info.get("target_attribute", "—")],
        ]
        mt = Table(meta, colWidths=[3.5 * cm, 6 * cm, 2.5 * cm, 5.5 * cm])
        mt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, white]),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ]))
        story.extend([mt, Spacer(1, 0.5 * cm)])

        # ── Verdict
        vt = Table([[Paragraph(f"BIAS STATUS: {severity}", verdict_s)]], colWidths=["100%"])
        vt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), sev_color),
            ("PADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(vt)
        story.append(Spacer(1, 0.3 * cm))
        if affected.get("narrative"):
            story.append(Paragraph(
                f'"{affected["narrative"]}"',
                ParagraphStyle("QQ", parent=styles["Normal"], fontSize=10,
                               fontName="Helvetica-BoldOblique",
                               textColor=sev_color, alignment=TA_CENTER)
            ))
        story.append(Spacer(1, 0.4 * cm))

        # ── 1. Key Metrics
        story.append(Paragraph("1. Key Fairness Metrics", section_s))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))

        di = metrics.get("disparate_impact", 0)
        dp = metrics.get("demographic_parity_diff", 0)

        def mrow(name, val, thresh, passes, desc):
            sc = GREEN if passes else RED
            return [name, f"{val:.4f}", thresh,
                    Paragraph("PASS" if passes else "FAIL",
                               ParagraphStyle("St", parent=styles["Normal"], fontSize=9,
                                              textColor=sc, fontName="Helvetica-Bold")),
                    desc]

        mdata = [
            ["Metric", "Value", "Threshold", "Status", "Description"],
            mrow("Disparate Impact Ratio", di, "≥ 0.80 (EEOC)", di >= 0.8, "80% rule compliance"),
            mrow("Demographic Parity Diff", dp, "< 0.10", dp < 0.1, "Selection rate gap"),
            mrow("Four-Fifths Rule", di, "≥ 0.80", metrics.get("passes_80_rule", False),
                 "EEOC employment standard"),
        ]

        # Add advanced metrics if present
        eo_diff = metrics.get("equal_opportunity_diff")
        pp_diff = metrics.get("predictive_parity_diff")
        if eo_diff is not None:
            mdata.append(mrow("Equal Opportunity Diff", eo_diff, "< 0.10",
                               eo_diff < 0.1, "TPR gap across groups"))
        if pp_diff is not None:
            mdata.append(mrow("Predictive Parity Diff", pp_diff, "< 0.10",
                               pp_diff < 0.1, "PPV gap across groups"))

        mt2 = Table(mdata, colWidths=[4 * cm, 2 * cm, 3.5 * cm, 2 * cm, 6 * cm])
        mt2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([mt2, Spacer(1, 0.4 * cm)])

        # ── 2. Group Statistics
        story.append(Paragraph("2. Group Selection Statistics", section_s))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))

        mean_rate = metrics.get("mean_selection_rate", 0)
        gdata = [["Group", "Sample Size", "Selected", "Selection Rate", "vs Mean", "Risk"]]
        for grp, st in group_stats.items():
            rate = st.get("selection_rate", 0)
            vs = rate - mean_rate
            is_unpriv = grp == metrics.get("unprivileged_group", "")
            flag = "LOW" if is_unpriv and dp > 0.1 else ("OK" if rate >= mean_rate else "MONITOR")
            if st.get("small_sample_warning"):
                flag += " *"
            gdata.append([
                grp,
                str(st.get("count", 0)),
                str(st.get("selected", st.get("count", 0))),
                f"{rate * 100:.1f}%",
                f"{'+' if vs >= 0 else ''}{vs * 100:.1f}%",
                flag,
            ])

        gt = Table(gdata, colWidths=[3 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 2.5 * cm, 4 * cm])
        gt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([gt, Spacer(1, 0.4 * cm)])

        # ── 3. Proxy Feature Risk (NEW — was missing from v1)
        if feature_importance:
            story.append(Paragraph("3. Proxy Feature Risk Analysis", section_s))
            story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
            story.append(Paragraph(
                "Features correlated with the sensitive attribute may act as proxies, "
                "reintroducing bias even when the sensitive attribute itself is excluded. "
                "Numeric correlation: Pearson |r|. Categorical: Cramér's V.",
                body_s,
            ))

            pfdata = [["Feature", "Correlation", "Method", "Risk Level"]]
            for feat in feature_importance[:8]:
                rc = RED if feat["risk_level"] == "HIGH" else (
                    YELLOW if feat["risk_level"] == "MEDIUM" else GREEN)
                pfdata.append([
                    feat["feature"],
                    f"{feat['correlation_with_sensitive']:.3f}",
                    feat.get("method", "pearson"),
                    Paragraph(feat["risk_level"],
                               ParagraphStyle("RL", parent=styles["Normal"], fontSize=9,
                                              textColor=rc, fontName="Helvetica-Bold")),
                ])

            pft = Table(pfdata, colWidths=[5 * cm, 3 * cm, 3.5 * cm, 6 * cm])
            pft.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([pft, Spacer(1, 0.4 * cm)])

        # Section number offset if proxy section present
        sec_offset = 1 if feature_importance else 0

        # ── AI Risk Assessment
        ai_exp = analysis.get("ai_explanation", "")
        if ai_exp:
            story.append(Paragraph(f"{3 + sec_offset}. AI Risk Assessment", section_s))
            story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
            story.append(Paragraph(ai_exp, body_s))
            story.append(Spacer(1, 0.3 * cm))

        # ── Remediation Results
        if fixed:
            story.append(Paragraph(f"{4 + sec_offset}. Remediation Results (Post-Fix)", section_s))
            story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))

            improvement = fixed.get("improvement", {})
            fixed_m = fixed.get("fixed_metrics", {})
            reduction = improvement.get("overall_bias_reduction_pct", 0)
            acc_delta = fixed.get("accuracy_delta")
            acc_str = f"{acc_delta:+.3f}" if acc_delta is not None else "N/A (label-only dataset)"

            story.append(Paragraph(
                f"Algorithm: <b>{fixed.get('algorithm', 'Reweighing')}</b> | "
                f"Bias Reduced: <b>{reduction:.0f}%</b> | "
                f"EEOC Compliant: <b>{'YES' if improvement.get('eeoc_compliant') else 'NO'}</b> | "
                f"Accuracy Delta: <b>{acc_str}</b>",
                body_s,
            ))

            if fixed.get("persistent_bias_warning"):
                story.append(Paragraph(
                    f"⚠️ {fixed['persistent_bias_warning']}", body_s
                ))

            fix_data = [
                ["Metric", "Before", "After", "Change", "Status"],
                ["Disparate Impact", f"{di:.4f}", f"{fixed_m.get('disparate_impact', 0):.4f}",
                 f"{improvement.get('disparate_impact_improvement', 0):+.1f}%",
                 "IMPROVED" if fixed_m.get("disparate_impact", 0) > di else "UNCHANGED"],
                ["DP Difference", f"{dp:.4f}", f"{fixed_m.get('demographic_parity_diff', 0):.4f}",
                 f"{improvement.get('demographic_parity_improvement', 0):+.1f}%", "REDUCED"],
                ["EEOC 80% Rule",
                 "FAIL" if di < 0.8 else "PASS",
                 "PASS" if fixed_m.get("passes_80_rule") else "FAIL",
                 "—",
                 "NOW COMPLIANT" if improvement.get("eeoc_compliant") else "CHECK"],
            ]
            fxt = Table(fix_data, colWidths=[4 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 5.5 * cm])
            fxt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(fxt)

            fix_ai = fixed.get("ai_explanation", "")
            if fix_ai:
                story.extend([Spacer(1, 0.3 * cm), Paragraph(fix_ai, body_s)])
            story.append(Spacer(1, 0.3 * cm))

        # ── Recommendations
        story.append(Paragraph(f"{5 + sec_offset}. Recommendations", section_s))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY))
        for i, rec in enumerate(self._get_recommendations(severity, metrics, fixed), 1):
            story.append(Paragraph(f"{i}. {rec}", body_s))
        story.append(Spacer(1, 0.5 * cm))

        # ── Footer
        story.append(HRFlowable(width="100%", thickness=1, color=MED_GRAY))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            f"Generated by FairLens AI v2.0 | {now} | "
            "Audit purposes only. Consult legal counsel for compliance decisions. | fairlens.ai",
            small_s,
        ))

        doc.build(story)
        return tmp.name

    @staticmethod
    def _get_recommendations(severity: str, metrics: Dict, fixed: Any) -> list:
        di = metrics.get("disparate_impact", 1.0)
        dp = metrics.get("demographic_parity_diff", 0)
        recs = []

        if severity == "SEVERE":
            recs.append(
                "DO NOT DEPLOY: Bias levels exceed EEOC thresholds. "
                "Immediate remediation required before any production use."
            )
            recs.append("Apply the FairLens debiasing fix and re-analyze before proceeding.")
        elif severity == "MODERATE":
            recs.append(
                "CONDITIONAL DEPLOYMENT: Apply debiasing fix and document "
                "remediation steps for legal protection."
            )

        if di < 0.8:
            recs.append(
                f"Disparate Impact ({di:.2f}) is below the EEOC 80% threshold. "
                "This violates US employment law standards for automated hiring systems."
            )
        if dp > 0.1:
            recs.append(
                f"Reduce the {dp*100:.1f}% selection rate gap through bias-aware training "
                "or post-processing threshold optimisation."
            )

        recs.append(
            "Establish quarterly bias monitoring with automated alerts "
            "when Disparate Impact drops below 0.85."
        )
        recs.append(
            "Review proxy features (correlated with sensitive attributes) "
            "and consider removing or transforming them."
        )
        recs.append(
            "Maintain this audit report as documentation of due diligence for "
            "regulatory compliance (EU AI Act, EEOC, NYC Local Law 144)."
        )

        if fixed and fixed.get("improvement", {}).get("eeoc_compliant"):
            recs.append(
                "After applying the debiasing fix, the system meets EEOC standards. "
                "Continue monitoring in production."
            )

        if fixed and fixed.get("persistent_bias_warning"):
            recs.append(
                "Automated debiasing did not achieve DI >= 0.80. "
                "Domain-specific intervention and legal review are required before deployment."
            )

        return recs
