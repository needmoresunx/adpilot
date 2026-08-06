from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from adpilot.critic.critique import CritiqueReport
from adpilot.identity.card import IdentityCard
from adpilot.planner.schema import ShotPlan
from adpilot.utils.json_io import write_json


MINIMUM_REMBG_VISUAL_SCORE = 50


def _alpha_mask(image: Image.Image) -> np.ndarray | None:
    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 3] > 16
    if not np.any(alpha):
        return None
    ys, xs = np.where(alpha)
    return alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _normalized_mask(mask: np.ndarray, canvas_size: int = 128) -> np.ndarray:
    source = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    fitted = ImageOps.contain(source, (canvas_size, canvas_size), Image.Resampling.NEAREST)
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    canvas.paste(fitted, ((canvas_size - fitted.width) // 2, (canvas_size - fitted.height) // 2))
    return np.asarray(canvas) > 127


def _masked_mean_rgb(image: Image.Image) -> np.ndarray | None:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3] > 16
    if not np.any(alpha):
        return None
    return rgba[:, :, :3][alpha].mean(axis=0)


def _visual_identity_metrics(reference: Image.Image, candidate: Image.Image) -> dict[str, float] | None:
    """Compute transparent, front-view visual similarity from two product cutouts."""
    reference_mask = _alpha_mask(reference)
    candidate_mask = _alpha_mask(candidate)
    reference_rgb = _masked_mean_rgb(reference)
    candidate_rgb = _masked_mean_rgb(candidate)
    if reference_mask is None or candidate_mask is None or reference_rgb is None or candidate_rgb is None:
        return None

    normalized_reference = _normalized_mask(reference_mask)
    normalized_candidate = _normalized_mask(candidate_mask)
    union = np.logical_or(normalized_reference, normalized_candidate).sum()
    silhouette_iou = float(np.logical_and(normalized_reference, normalized_candidate).sum() / max(union, 1))
    reference_ratio = reference_mask.shape[1] / max(reference_mask.shape[0], 1)
    candidate_ratio = candidate_mask.shape[1] / max(candidate_mask.shape[0], 1)
    aspect_similarity = max(0.0, 1.0 - abs(np.log(max(candidate_ratio, 1e-6) / max(reference_ratio, 1e-6))) / np.log(2.0))
    color_delta = float(np.linalg.norm(reference_rgb - candidate_rgb))
    color_similarity = max(0.0, 1.0 - color_delta / (255.0 * np.sqrt(3.0)))
    score = round(100.0 * (0.60 * silhouette_iou + 0.25 * aspect_similarity + 0.15 * color_similarity))
    return {
        "visual_score": int(score),
        "silhouette_iou": float(round(silhouette_iou, 4)),
        "aspect_similarity": float(round(aspect_similarity, 4)),
        "color_similarity": float(round(color_similarity, 4)),
        "color_delta": float(round(color_delta, 2)),
    }


def _candidate_cutout(image: Image.Image, session: Any) -> Image.Image:
    from rembg import remove

    return remove(image.convert("RGBA"), session=session).convert("RGBA")


def _grabcut_candidate_cutout(image: Image.Image) -> Image.Image | None:
    """Estimate the foreground in the known product ROI without another model.

    FLUX receives the product's planned position, so the product is expected in
    the central part of this crop. GrabCut is only a fallback when the optional
    rembg weight is absent; its output is recorded so it is never mistaken for
    a ground-truth mask.
    """
    try:
        import cv2
    except ImportError:
        return None
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if width < 12 or height < 12:
        return None
    margin_x = max(2, round(width * 0.12))
    margin_y = max(2, round(height * 0.08))
    rectangle = (margin_x, margin_y, max(2, width - 2 * margin_x), max(2, height - 2 * margin_y))
    mask = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rectangle, background, foreground, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if int((alpha > 16).sum()) < max(32, round(width * height * 0.01)):
        return None
    rgba = np.dstack((rgb, alpha))
    return Image.fromarray(rgba, "RGBA")


def _corner_background_cutout(image: Image.Image) -> Image.Image | None:
    """Dependency-free matte for a product centered in its planned ROI.

    It estimates background color from the ROI corners, just like the original
    reference-photo cutout fallback. This is intentionally reported as a
    heuristic rather than presented as a learned segmentation result.
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    height, width = rgba.shape[:2]
    if width < 8 or height < 8:
        return None
    span_y, span_x = max(1, height // 10), max(1, width // 10)
    corners = (
        rgba[:span_y, :span_x, :3],
        rgba[:span_y, -span_x:, :3],
        rgba[-span_y:, :span_x, :3],
        rgba[-span_y:, -span_x:, :3],
    )
    corner_pixels = np.concatenate([patch.reshape(-1, 3) for patch in corners], axis=0).astype(np.float32)
    background = np.median(corner_pixels, axis=0)
    corner_distance = np.linalg.norm(corner_pixels - background, axis=1)
    median_deviation = float(np.median(np.abs(corner_distance - np.median(corner_distance))))
    threshold = max(22.0, float(np.median(corner_distance) + 3.0 * median_deviation))
    distance = np.linalg.norm(rgba[:, :, :3].astype(np.float32) - background, axis=2)
    alpha = np.where(distance > threshold, 255, 0).astype(np.uint8)
    foreground_pixels = int((alpha > 16).sum())
    if foreground_pixels < max(32, round(width * height * 0.01)):
        return None
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def _segmentation_session() -> Any | None:
    """Reuse the already-required product-segmentation weight; never download at audit time."""
    try:
        # rembg forwards OMP_NUM_THREADS into ONNX Runtime SessionOptions. An
        # explicit value prevents ORT from pinning threads outside an HPC CPU set.
        os.environ["OMP_NUM_THREADS"] = os.environ.get("ADPILOT_ONNX_THREADS", "1")
        from rembg import new_session
        from adpilot.identity.builder import cached_rembg_model_path

        if cached_rembg_model_path("birefnet-general") is None:
            return None
        return new_session("birefnet-general")
    except Exception:
        return None


def _expected_roi(size: tuple[int, int], shot: ShotPlan, aspect_ratio: float) -> tuple[int, int, int, int]:
    width, height = size
    product_width = max(1, round(width * min(max(float(shot.product_scale), 0.18), 0.62)))
    product_height = min(max(1, round(product_width / max(aspect_ratio, 0.05))), round(height * 0.82))
    product_width = max(1, round(product_height * aspect_ratio))
    center_x = {"left": 0.28, "center": 0.50, "right": 0.72}.get(shot.product_position, 0.50) * width
    left = max(0, min(width - product_width, round(center_x - product_width / 2)))
    top = max(0, min(height - product_height, round(height * 0.88 - product_height)))
    pad_x = round(product_width * 0.22)
    pad_y = round(product_height * 0.22)
    return max(0, left - pad_x), max(0, top - pad_y), min(width, left + product_width + pad_x), min(height, top + product_height + pad_y)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (242, 242, 242))
    fitted = ImageOps.contain(image.convert("RGB"), size)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _write_board(
    reference: Image.Image,
    candidate: Image.Image,
    roi: tuple[int, int, int, int],
    report: CritiqueReport,
    output_path: Path,
) -> None:
    panel = (320, 240)
    board = Image.new("RGB", (panel[0] * 3, panel[1] + 72), "white")
    draw = ImageDraw.Draw(board)
    marked = candidate.convert("RGB").copy()
    draw_candidate = ImageDraw.Draw(marked)
    draw_candidate.rectangle(roi, outline=(220, 30, 30), width=max(2, candidate.width // 240))
    images = (reference, marked, candidate.crop(roi))
    labels = ("reference", "generated frame", "planned product region")
    for index, (image, label) in enumerate(zip(images, labels)):
        board.paste(_fit(image, panel), (index * panel[0], 34))
        draw.text((index * panel[0] + 8, 10), label, fill="black")
    checks = report.identity_checks
    summary = " | ".join(f"{name.replace('_match', '')}: {checks.get(name, 'unknown')}" for name in ("silhouette_match", "component_match", "color_match"))
    metric = report.identity_evidence.get("visual_metric", {})
    if metric.get("available"):
        summary = f"visual score: {metric['visual_score']} | {summary}"
    draw.text((8, panel[1] + 42), summary, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)


def build_keyframe_identity_evidence(
    identity_card: IdentityCard,
    shots: list[ShotPlan],
    candidate_paths: list[list[Path]],
    reports: list[list[CritiqueReport]],
    output_dir: Path,
    enforce_visual_gate: bool = False,
) -> list[list[dict[str, Any]]]:
    """Create inspectable reference/frame/crop boards without adding another model."""
    reference = Image.open(identity_card.cutout).convert("RGBA")
    session = _segmentation_session()
    groups: list[list[dict[str, Any]]] = []
    for shot, paths, shot_reports in zip(shots, candidate_paths, reports):
        group: list[dict[str, Any]] = []
        for index, (candidate_path, report) in enumerate(zip(paths, shot_reports), start=1):
            candidate = Image.open(candidate_path).convert("RGB")
            roi = _expected_roi(candidate.size, shot, identity_card.aspect_ratio)
            candidate_dir = output_dir / shot.shot_id / f"candidate_{index:02d}"
            crop_path = candidate_dir / "product_region.png"
            board_path = candidate_dir / "identity_evidence.png"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            product_region = candidate.crop(roi)
            product_region.save(crop_path)
            metric: dict[str, Any] = {"available": False, "reason": "candidate_segmentation_unavailable"}
            generated_cutout = None
            metric_source = None
            if session is not None:
                try:
                    generated_cutout = _candidate_cutout(product_region, session)
                    metric_source = "rembg_birefnet"
                except Exception as exc:
                    metric = {"available": False, "reason": f"candidate_segmentation_failed:{type(exc).__name__}"}
            if generated_cutout is None:
                generated_cutout = _grabcut_candidate_cutout(product_region)
                metric_source = "opencv_grabcut" if generated_cutout is not None else None
            if generated_cutout is None:
                generated_cutout = _corner_background_cutout(product_region)
                metric_source = "corner_background_heuristic" if generated_cutout is not None else None
            if generated_cutout is not None:
                cutout_path = candidate_dir / "candidate_product_cutout.png"
                generated_cutout.save(cutout_path)
                values = _visual_identity_metrics(reference, generated_cutout)
                if values is not None:
                    metric = {"available": True, **values, "source": metric_source, "candidate_cutout": str(cutout_path)}
                    report.shape_score = values["silhouette_iou"]
                    report.color_delta = values["color_delta"]
                    report.identity_audit_score = values["visual_score"] if report.passed else min(values["visual_score"], 49)
                    if (
                        enforce_visual_gate
                        and metric_source == "rembg_birefnet"
                        and report.passed
                        and values["visual_score"] < MINIMUM_REMBG_VISUAL_SCORE
                    ):
                        report.failure_reasons.append("visual_identity_below_threshold")
                        report.passed = False
                        report.repair_instruction = "restore the reference silhouette and components"
                else:
                    metric = {"available": False, "reason": "empty_product_mask", "source": metric_source}
            report.identity_evidence = {"visual_metric": metric}
            _write_board(reference, candidate, roi, report, board_path)
            group.append(
                {
                    "candidate": index,
                    "expected_roi": list(roi),
                    "product_region": str(crop_path),
                    "evidence_image": str(board_path),
                    "identity_checks": dict(report.identity_checks),
                    "visual_metric": metric,
                    "verdict": report.identity_verdict,
                    "evidence": report.evidence,
                }
            )
        groups.append(group)
    write_json(output_dir / "identity_evidence.json", {shot.shot_id: group for shot, group in zip(shots, groups)})
    return groups
