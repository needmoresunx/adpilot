from __future__ import annotations

import html
import json
from pathlib import Path

from adpilot.critic.critique import CritiqueReport
from adpilot.identity.card import IdentityCard


def _rel(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    return path.resolve().relative_to(base.resolve()).as_posix()


def write_html_report(
    run_dir: Path,
    identity_card: IdentityCard,
    final_frames: list[Path],
    reports: list[CritiqueReport],
    repair_log: list[dict],
    storyboard_path: Path,
    preview_path: Path | None,
    generation_metadata: dict | None = None,
    video_metadata: dict | None = None,
) -> Path:
    rows = []
    for frame, report in zip(final_frames, reports):
        reasons = ", ".join(report.failure_reasons) if report.failure_reasons else "none"
        rows.append(
            f"""
            <article class="shot">
              <img src="{html.escape(_rel(frame, run_dir) or '')}" />
              <div>
                <h3>{html.escape(report.shot_id)}: {'PASS' if report.passed else 'FAIL'}</h3>
                <p>scale: {report.product_scale}</p>
                <p>color_delta: {report.color_delta}</p>
                <p>shape_score: {report.shape_score}</p>
                <p>logo_area_ratio: {report.logo_area_ratio}</p>
                <p>product_bbox: {report.product_bbox}</p>
                <p>logo_bbox_in_frame: {report.logo_bbox_in_frame}</p>
                <p>ocr_available: {report.ocr_available}</p>
                <p>ocr_text: {html.escape(report.ocr_text or 'none')}</p>
                <p>failure_reasons: {html.escape(reasons)}</p>
              </div>
            </article>
            """
        )

    log_items = "\n".join(f"<li><code>{html.escape(str(item))}</code></li>" for item in repair_log) or "<li>No repair needed.</li>"
    preview_html = ""
    if preview_path:
        preview_html = f'<video controls src="{html.escape(_rel(preview_path, run_dir) or "")}"></video>'
    generation_json = html.escape(json.dumps(generation_metadata or {}, indent=2))
    video_json = html.escape(json.dumps(video_metadata or {}, indent=2))
    generation_fallback = bool((generation_metadata or {}).get("used_fallback"))
    video_fallback = bool((video_metadata or {}).get("used_fallback"))
    video_name = (video_metadata or {}).get("name")
    video_label = {
        "wan_i2v": "Wan I2V Generated Video",
        "proxy": "Proxy Preview",
    }.get(video_name, "Video Preview")
    product_brief_json = html.escape(json.dumps(identity_card.product_brief, indent=2))
    status_text = "PASS"
    if generation_fallback or video_fallback:
        status_text = "FALLBACK USED"

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
    .storyboard {{ max-width: 900px; width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
    video {{ width: 320px; max-width: 100%; border-radius: 6px; border: 1px solid #ddd; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>AdPilot Identity Audit Report</h1>
  <section class="panel">
    <h2>Run Status</h2>
    <p><span class="status {'warn' if generation_fallback or video_fallback else ''}">{status_text}</span></p>
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
  </section>
  <section class="panel">
    <h2>Shot Critique</h2>
    {''.join(rows)}
  </section>
  <section class="panel">
    <h2>Repair Log</h2>
    <ul>{log_items}</ul>
  </section>
</body>
</html>
"""
    output = run_dir / "report.html"
    output.write_text(page, encoding="utf-8")
    return output
