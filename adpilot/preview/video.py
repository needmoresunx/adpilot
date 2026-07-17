from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def export_concat_preview_if_available(frame_paths: list[Path], output_path: Path, seconds_per_frame: int = 3) -> Path | None:
    if not shutil.which("ffmpeg") or not frame_paths:
        return None
    list_path = output_path.with_suffix(".txt")
    lines = []
    for frame in frame_paths:
        lines.append(f"file '{frame.resolve()}'")
        lines.append(f"duration {seconds_per_frame}")
    lines.append(f"file '{frame_paths[-1].resolve()}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-vf", "format=yuv420p", str(output_path)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    return output_path if output_path.exists() else None


def export_motion_preview_if_available(
    frame_paths: list[Path],
    output_path: Path,
    seconds_per_frame: int = 3,
    fps: int = 12,
    size: tuple[int, int] = (540, 960),
) -> Path | None:
    if not shutil.which("ffmpeg") or not frame_paths:
        return None
    cmd = ["ffmpeg", "-y"]
    for frame in frame_paths:
        cmd.extend(["-loop", "1", "-t", str(seconds_per_frame), "-i", str(frame)])

    filters: list[str] = []
    labels: list[str] = []
    frames_per_shot = seconds_per_frame * fps
    width, height = size
    for index in range(len(frame_paths)):
        label = f"v{index}"
        labels.append(f"[{label}]")
        filters.append(
            f"[{index}:v]"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            "zoompan=z='min(zoom+0.0012,1.08)':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            f"d={frames_per_shot}:s={width}x{height}:fps={fps},"
            f"setsar=1[{label}]"
        )
    filters.append(f"{''.join(labels)}concat=n={len(frame_paths)}:v=1:a=0,format=yuv420p[outv]")
    cmd.extend(["-filter_complex", ";".join(filters), "-map", "[outv]", "-r", str(fps), str(output_path)])
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if output_path.exists():
            return output_path
    except subprocess.TimeoutExpired:
        pass
    return export_concat_preview_if_available(frame_paths, output_path, seconds_per_frame)

