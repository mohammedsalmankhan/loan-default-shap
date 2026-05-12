"""PDF decision report generation for individual loan applicants.

Produces a one-page audit-friendly report containing:
    - The model's decision and predicted default probability
    - The applicant's input features
    - The top SHAP-driven reasons that drove the decision
    - A "reasons" sentence in plain English for customer communication
"""
from __future__ import annotations

import io
from typing import List, Tuple

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors


def build_report(
    applicant_features: pd.DataFrame,
    decision: str,
    probability: float,
    top_reasons: List[Tuple[str, float]],
) -> bytes:
    """Build a PDF report and return the raw bytes (ready for st.download_button).

    Args:
        applicant_features: Single-row DataFrame with the applicant's features.
        decision: "APPROVE" or "REJECT".
        probability: Predicted default probability in [0, 1].
        top_reasons: List of (feature_name, mean_abs_shap_value) tuples.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    subheading_style = ParagraphStyle(
        "Subheading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=6,
    )
    body_style = styles["BodyText"]

    decision_colour = colors.green if decision == "APPROVE" else colors.red

    story = []

    story.append(Paragraph("Loan Decision Report", heading_style))
    story.append(Spacer(1, 0.3 * cm))

    decision_para = Paragraph(
        f'<b>Decision:</b> <font color="{decision_colour.hexval()}">{decision}</font>',
        body_style,
    )
    story.append(decision_para)
    story.append(Paragraph(
        f"<b>Predicted default probability:</b> {probability:.1%}", body_style,
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Applicant details", subheading_style))
    rows = [["Feature", "Value"]]
    for col in applicant_features.columns:
        value = applicant_features[col].values[0]
        rows.append([col, str(value)])
    table = Table(rows, colWidths=[6 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Top reasons driving this decision", subheading_style))
    reasons_rows = [["Feature", "Impact (mean |SHAP|)"]]
    for name, impact in top_reasons:
        reasons_rows.append([name, f"{impact:.4f}"])
    reasons_table = Table(reasons_rows, colWidths=[8 * cm, 6 * cm])
    reasons_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(reasons_table)
    story.append(Spacer(1, 0.4 * cm))

    if len(top_reasons) >= 2:
        explanation = (
            f"The loan was {'approved' if decision == 'APPROVE' else 'rejected'} "
            f"primarily because of <b>{top_reasons[0][0]}</b> and "
            f"<b>{top_reasons[1][0]}</b>."
        )
        story.append(Paragraph(explanation, body_style))

    story.append(Spacer(1, 0.6 * cm))
    disclaimer = (
        '<font size="8" color="grey">This report was produced by an automated '
        "decision system. Under UK consumer credit and equality law, applicants "
        "have the right to request a manual review of automated lending decisions. "
        "SHAP attributions are estimates and may be unstable for rare or extreme "
        "applicant profiles.</font>"
    )
    story.append(Paragraph(disclaimer, body_style))

    doc.build(story)
    return buffer.getvalue()
