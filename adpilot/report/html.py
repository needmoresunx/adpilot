from __future__ import annotations

import html
import json
import os
from pathlib import Path

from adpilot.critic.critique import CritiqueReport
from adpilot.identity.card import IdentityCard


def _rel(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    return Path(os.path.relpath(path.resolve(), start=base.resolve())).as_posix()


def write_html_report(
    run_dir: Path,
    identity_card: IdentityCard,
    final_frames: list[Path],
    reports: list[CritiqueReport],
    storyboard_path: Path,
    preview_path: Path | None,
    generation_metadata: dict | None = None,
    video_metadata: dict | None = None,
    video_reports: list[CritiqueReport] | None = None,
    planner_metadata: dict | None = None,
    repair_log: list[dict] | None = None,
) -> Path:
    rows = []
    for frame, report in zip(final_frames, reports):
        reasons = ", ".join(report.failure_reasons) if report.failure_reasons else "none"
        evidence_path = report.identity_evidence.get("evidence_image")
        visual_metric = report.identity_evidence.get("visual_metric", {})
        evidence_html = ""
        if evidence_path:
            evidence_html = (
                f'<img class="evidence" src="{html.escape(_rel(Path(evidence_path), run_dir) or "")}" '
                'alt="identity evidence" />'
            )
        rows.append(
            f"""
            <article class="shot">
              <img src="{html.escape(_rel(frame, run_dir) or '')}" />
              <div>
                <h3>{html.escape(report.shot_id)}: {'PASS' if report.passed else 'FAIL'}</h3>
                <p>scale: {report.product_scale}</p>
                <p>color_delta: {report.color_delta}</p>
                <p>logo_area_ratio: {report.logo_area_ratio}</p>
                <p>product_bbox: {report.product_bbox}</p>
                <p>logo_bbox_in_frame: {report.logo_bbox_in_frame}</p>
                <p>ocr_available: {report.ocr_available}</p>
                <p>ocr_text: {html.escape(report.ocr_text or 'none')}</p>
                <p>critic: {html.escape(report.critic_name)}</p>
                <p>identity_verdict: {html.escape(report.identity_verdict or 'not evaluated')}</p>
                <p>visual_identity_score: {report.identity_audit_score}</p>
                <p>visual_metric: {html.escape(json.dumps(visual_metric, ensure_ascii=True))}</p>
                <p>identity_checks: {html.escape(json.dumps(report.identity_checks, ensure_ascii=True))}</p>
                {evidence_html}
                <p>product_visible: {report.product_visible}</p>
                <p>product_count: {report.product_count}</p>
                <p>label_readability: {html.escape(report.label_readability or 'not evaluated')}</p>
                <p>visual_drift: {html.escape(', '.join(report.visual_drift) or 'none')}</p>
                <p>recommended_action: {html.escape(report.repair_instruction or 'none')}</p>
                <p>evidence: {html.escape(report.evidence or 'none')}</p>
                <p>failure_reasons: {html.escape(reasons)}</p>
              </div>
            </article>
            """
        )

    video_rows = []
    for report in video_reports or []:
        reasons = ", ".join(report.failure_reasons) if report.failure_reasons else "none"
        video_rows.append(
            f"<li><strong>{html.escape(report.shot_id)}</strong>: {'PASS' if report.passed else 'FAIL'}; "
            f"identity_verdict={html.escape(report.identity_verdict or 'not evaluated')}; temporal_consistency="
            f"{html.escape(report.temporal_consistency or 'not evaluated')}; "
            f"failure_reasons={html.escape(reasons)}</li>"
        )
    repair_rows = []
    for item in repair_log or []:
        repair_rows.append(
            f"<li><strong>{html.escape(str(item.get('stage', 'unknown')))} / "
            f"{html.escape(str(item.get('shot_id', 'unknown')))}</strong>: "
            f"{'repaired' if item.get('repaired') else 'still failed'}; "
            f"reason={html.escape(', '.join(item.get('failure_reasons', [])) or 'none')}</li>"
        )
    preview_html = ""
    if preview_path:
        preview_html = f'<video controls src="{html.escape(_rel(preview_path, run_dir) or "")}"></video>'
    generation_json = html.escape(json.dumps(generation_metadata or {}, indent=2))
    video_json = html.escape(json.dumps(video_metadata or {}, indent=2))
    planner_json = html.escape(json.dumps(planner_metadata or {}, indent=2))
    generation_fallback = bool((generation_metadata or {}).get("used_fallback"))
    video_fallback = bool((video_metadata or {}).get("used_fallback"))
    keyframe_audit_failed = any(not report.passed for report in reports)
    video_audit_failed = any(not report.passed for report in (video_reports or []))
    video_name = (video_metadata or {}).get("name")
    video_label = {
        "wan_i2v": "Wan I2V Generated Video",
    }.get(video_name, "Video Preview")
    product_brief_json = html.escape(json.dumps(identity_card.product_brief, indent=2))
    status_text = "PASS"
    if generation_fallback or video_fallback:
        status_text = "FALLBACK USED"
    elif keyframe_audit_failed or video_audit_failed:
        status_text = "IDENTITY CHECK FAILED"

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AdPilot Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background: #f6f6f3; color: #1f2933; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .panel {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 18px 0; }}
    .status {{ display: inline-block; padding: 6px 10px; border-radius: 6px; background: #e5e7eb; font-weight: 700; }}
    .status.warn {{ background: #fef3c7; }}
    .shot {{ display: grid; grid-template-columns: 180px 1fr; gap: 18px; align-items: start; margin: 18px 0; }}
    .shot img {{ width: 180px; border-radius: 6px; border: 1px solid #ddd; }}
    .shot img.evidence {{ width: min(100%, 640px); margin-top: 8px; }}
    .storyboard {{ max-width: 900px; width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
    video {{ width: 320px; max-width: 100%; border-radius: 6px; border: 1px solid #ddd; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>AdPilot Identity Audit Report</h1>
  <section class="panel">
    <h2>Run Status</h2>
    <p><span class="status {'warn' if generation_fallback or video_fallback or keyframe_audit_failed or video_audit_failed else ''}">{status_text}</span></p>
    <p>generation_backend: {html.escape(str((generation_metadata or {}).get('name', 'unknown')))}</p>
    <p>generation_used_fallback: {generation_fallback}</p>
    <p>video_backend: {html.escape(str((video_metadata or {}).get('name', 'unknown')))}</p>
    <p>video_used_fallback: {video_fallback}</p>
  </section>
  <section class="panel">
    <h2>Product Identity</h2>
    <p>brand: {html.escape(identity_card.brand)}</p>
    <p>dominant_rgb: {identity_card.dominant_rgb}</p>
    <p>aspect_ratio: {identity_card.aspect_ratio}</p>
    <p>logo_bbox: {identity_card.logo_bbox}</p>
    <h3>Product Brief</h3>
    <pre><code>{product_brief_json}</code></pre>
  </section>
  <section class="panel">
    <h2>{html.escape(video_label)}</h2>
    {preview_html}
    <p><img class="storyboard" src="{html.escape(_rel(storyboard_path, run_dir) or '')}" /></p>
  </section>
  <section class="panel">
    <h2>Backend Metadata</h2>
    <h3>Generation</h3>
    <pre><code>{generation_json}</code></pre>
    <h3>Video</h3>
    <pre><code>{video_json}</code></pre>
    <h3>Planner</h3>
    <pre><code>{planner_json}</code></pre>
  </section>
  <section class="panel">
    <h2>Shot Critique</h2>
    {''.join(rows)}
  </section>
  <section class="panel">
    <h2>Video Frame Critique</h2>
    <ul>{''.join(video_rows) or '<li>Not evaluated.</li>'}</ul>
  </section>
  <section class="panel">
    <h2>Bounded Repair Loop</h2>
    <ul>{''.join(repair_rows) or '<li>No repair was needed.</li>'}</ul>
  </section>
</body>
</html>
"""
    output = run_dir / "report.html"
    output.write_text(page, encoding="utf-8")
    return output
