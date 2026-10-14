"""
MANZAR — PDF Report Generator
================================
Generates a professional police-style PDF report using xhtml2pdf.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

logger = logging.getLogger("manzar.report_pdf")


def _md_to_html(text: str) -> str:
    """Simple markdown-to-HTML for report text (bold, italic, lists, line breaks)."""
    if not text:
        return ""
    import html as _html
    import re
    s = _html.escape(str(text))
    # Bold
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'__(.+?)__', r'<b>\1</b>', s)
    # Italic
    s = re.sub(r'\*([^*\n]+?)\*', r'<i>\1</i>', s)
    s = re.sub(r'_([^_\n]+?)_', r'<i>\1</i>', s)
    # Unordered lists
    lines = s.split('\n')
    in_list = False
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                out.append('<ul>')
                in_list = True
            out.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append(line)
    if in_list:
        out.append('</ul>')
    s = '\n'.join(out)
    # Line breaks (but not inside lists)
    s = s.replace('\n', '<br/>')
    # Clean up br around lists
    s = re.sub(r'<br/>(<ul>)', r'\1', s)
    s = re.sub(r'(</ul>)<br/>', r'\1', s)
    return s


def generate_pdf(report: dict[str, Any], session: dict[str, Any]) -> bytes:
    """Generate a PDF from the report and return raw bytes."""
    from xhtml2pdf import pisa

    html_str = _build_html(report)
    buffer = BytesIO()
    status = pisa.CreatePDF(html_str, dest=buffer)
    if status.err:
        raise RuntimeError(f"PDF generation failed with {status.err} errors")
    pdf_bytes = buffer.getvalue()
    logger.info(f"PDF: {len(pdf_bytes)} bytes for {report.get('case_header', {}).get('case_number', '?')}")
    return pdf_bytes


def _build_html(report: dict[str, Any]) -> str:
    header = report.get("case_header", {})
    summary = report.get("summary", "")
    findings = report.get("evidence_findings", [])
    forensic = report.get("forensic_analysis", {})
    cross = report.get("cross_angle_verification", {})
    coverage = report.get("coverage", {})
    recon = report.get("reconstruction", {})
    custody = report.get("chain_of_custody", [])
    timeline = report.get("investigation_timeline", [])
    recommendations = report.get("recommendations", [])
    possible_cause = report.get("possible_cause", "")
    disclaimer = report.get("disclaimer", "")

    # Build findings rows
    findings_rows = ""
    for f in findings:
        sev_color = {"CRITICAL": "#cf1322", "HIGH": "#d46b08", "MEDIUM": "#0958d9", "LOW": "#389e0d"}.get(f.get("severity", ""), "#666")
        findings_rows += f"""
        <tr>
            <td><b>{f.get('label', '?')}</b></td>
            <td style="color:{sev_color}; font-weight:700;">{f.get('severity', '?')}</td>
            <td>{f.get('count', 0)}</td>
            <td>{f.get('max_confidence', 0) * 100:.0f}%</td>
            <td>{', '.join(f.get('angles', [])) or 'N/A'}</td>
            <td style="font-weight:700; color:{'#389e0d' if f.get('status')=='VERIFIED' else '#cf1322'};">{f.get('status', '?')}</td>
        </tr>"""

    if not findings_rows:
        findings_rows = '<tr><td colspan="6" align="center"><i>No evidence detected</i></td></tr>'

    # Build custody rows
    custody_rows = ""
    for c in custody:
        custody_rows += f"""
        <tr>
            <td>#{c.get('photo_number', 0)}</td>
            <td>{c.get('device', '?')}</td>
            <td>{c.get('angle', '?')}</td>
            <td>{c.get('uploaded_at', '')[:19].replace('T', ' ')}</td>
            <td style="font-family:monospace; font-size:8px;">{c.get('hash', 'N/A')}</td>
            <td>{c.get('detections', 0)}</td>
        </tr>"""

    # Build timeline rows
    timeline_rows = ""
    icons = {"start": "&#9654;", "phone": "&#9742;", "camera": "&#128247;", "check": "&#10004;", "verify": "&#9733;", "report": "&#128196;"}
    for t in timeline:
        icon = icons.get(t.get("icon", ""), "&#8226;")
        timeline_rows += f"""
        <tr>
            <td style="width:150px; font-family:monospace; font-size:9px;">{t.get('time', '')}</td>
            <td style="width:20px; text-align:center;">{icon}</td>
            <td>{t.get('detail', '')}</td>
        </tr>"""

    # Build forensic section
    forensic_items = ""
    forensic_labels = {
        "skid_mark_length": "Skid Mark Length",
        "speed_estimate": "Estimated Speed",
        "vehicle_damage_count": "Vehicle Damage Instances",
        "vehicle_damage_confidence": "Damage Detection Confidence",
        "damage_corroboration": "Damage Corroboration",
        "fault_indicator": "Fault Indicator",
        "debris_detected": "Debris Field",
        "impact_points": "Impact Points",
        "impact_status": "Impact Status",
        "scene_complexity": "Scene Complexity",
    }
    for key, val in forensic.items():
        label = forensic_labels.get(key, key.replace("_", " ").title())
        rendered = _md_to_html(val) if isinstance(val, str) else str(val)
        forensic_items += f"<tr><td><b>{label}</b></td><td>{rendered}</td></tr>"

    # Recommendations
    rec_html = ""
    for r in recommendations:
        rec_html += f"<li>{_md_to_html(r)}</li>"

    gps_str = ""
    if header.get("gps"):
        gps_str = f"GPS: {header['gps']['lat']:.6f}, {header['gps']['lng']:.6f}"

    cross_score = cross.get("overall_score", 0)
    cross_reliability = cross.get("reliability", "N/A")
    cross_color = {"HIGH": "#389e0d", "MODERATE": "#d46b08", "LOW": "#cf1322"}.get(cross_reliability, "#666")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: Helvetica, Arial, sans-serif;
        font-size: 10px;
        line-height: 1.4;
        color: #1a1a1a;
        margin: 15px;
    }}
    h1 {{
        font-size: 18px;
        color: #1a237e;
        border-bottom: 3px solid #1a237e;
        padding-bottom: 6px;
        margin-bottom: 4px;
    }}
    h2 {{
        font-size: 13px;
        color: #1a237e;
        margin-top: 16px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 2px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 6px 0;
    }}
    th {{
        background: #1a237e;
        color: white;
        padding: 4px 6px;
        text-align: left;
        font-size: 9px;
    }}
    td {{
        padding: 3px 6px;
        border-bottom: 1px solid #e0e0e0;
        font-size: 9px;
    }}
    .header-table {{ margin: 8px 0; }}
    .header-table td {{ border: none; padding: 2px 8px; }}
    .header-table .label {{ color: #666; font-weight: 700; font-size: 8px; text-transform: uppercase; }}
    .summary-box {{
        background: #e8eaf6;
        border-left: 3px solid #1a237e;
        padding: 8px 12px;
        margin: 8px 0;
        font-size: 10px;
    }}
    .disclaimer {{
        margin-top: 20px;
        padding: 8px;
        background: #fff3e0;
        border: 1px solid #ffcc80;
        font-size: 8px;
        color: #e65100;
    }}
    .footer {{
        margin-top: 20px;
        text-align: center;
        font-size: 8px;
        color: #999;
        border-top: 1px solid #ddd;
        padding-top: 8px;
    }}
    .stamp {{
        float: right;
        border: 2px solid #1a237e;
        padding: 4px 12px;
        font-weight: 700;
        color: #1a237e;
        font-size: 10px;
        transform: rotate(-5deg);
    }}
</style>
</head>
<body>

<h1>ACCIDENT RECONSTRUCTION REPORT</h1>
<div style="text-align:right; margin-top:-30px;"><div class="stamp">MANZAR AI</div></div>

<table class="header-table">
<tr>
    <td style="width:50%;"><span class="label">CASE NUMBER</span><br/><b>{header.get('case_number', 'N/A')}</b></td>
    <td><span class="label">SESSION ID</span><br/><b>{header.get('session_id', 'N/A')}</b></td>
</tr>
<tr>
    <td><span class="label">INVESTIGATING OFFICER</span><br/><b>{header.get('officer_name', 'N/A')}</b> (Badge: {header.get('officer_badge', 'N/A')})</td>
    <td><span class="label">LOCATION</span><br/><b>{header.get('location', 'N/A')}</b><br/>{gps_str}</td>
</tr>
<tr>
    <td><span class="label">DATE INITIATED</span><br/>{header.get('date_initiated', 'N/A')}</td>
    <td><span class="label">REPORT GENERATED</span><br/>{header.get('date_generated', 'N/A')}</td>
</tr>
</table>

<div class="summary-box"><b>EXECUTIVE SUMMARY:</b> {_md_to_html(summary)}</div>

<h2>POSSIBLE CAUSE OF ACCIDENT</h2>
<div class="summary-box" style="background:#f3e5f5; border-left:3px solid #7b1fa2;">{_md_to_html(possible_cause)}</div>

<h2>EVIDENCE FINDINGS</h2>
<table>
<tr><th>Evidence Type</th><th>Severity</th><th>Count</th><th>Confidence</th><th>Angles</th><th>Status</th></tr>
{findings_rows}
</table>

<h2>FORENSIC ANALYSIS</h2>
<table>{forensic_items or '<tr><td colspan="2"><i>No forensic analysis available</i></td></tr>'}</table>

<h2>CROSS-ANGLE VERIFICATION</h2>
<p><b>Overall Score:</b> {cross_score * 100:.0f}% | <b>Reliability:</b> <span style="color:{cross_color}; font-weight:700;">{cross_reliability}</span></p>

<h2>COVERAGE</h2>
<p><b>Status:</b> {coverage.get('status', 'unknown').upper()} | <b>Angles:</b> {coverage.get('coverage_pct', '0/4')} covered</p>

<h2>3D RECONSTRUCTION</h2>
<table>
<tr><td><b>Status</b></td><td>{recon.get('status', 'not started')}</td></tr>
<tr><td><b>Backend</b></td><td>{recon.get('backend', 'N/A')}</td></tr>
<tr><td><b>Mesh</b></td><td>{"Available" if recon.get("mesh_available") else "Not available"}</td></tr>
<tr><td><b>Point Cloud</b></td><td>{"Available" if recon.get("pointcloud_available") else "Not available"}</td></tr>
</table>

<h2>CHAIN OF CUSTODY</h2>
<table>
<tr><th>#</th><th>Device</th><th>Angle</th><th>Timestamp</th><th>SHA-256 Hash</th><th>Dets</th></tr>
{custody_rows}
</table>

<h2>INVESTIGATION TIMELINE</h2>
<table>{timeline_rows}</table>

<h2>RECOMMENDATIONS</h2>
<ol>{rec_html}</ol>

<div class="disclaimer"><b>DISCLAIMER:</b> {_md_to_html(disclaimer)}</div>

<div class="footer">
    MANZAR — AI-Powered Traffic Accident Reconstruction System<br/>
    This document is auto-generated. For official use, verify with investigating officer.
</div>

</body>
</html>"""
    return html
