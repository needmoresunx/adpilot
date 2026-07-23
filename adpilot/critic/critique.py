from __future__ import annotations

from dataclasses import asdict, dataclass, field

@dataclass
class CritiqueReport:
    shot_id: str
    passed: bool
    product_scale: float
    color_delta: float
    shape_score: float
    logo_area_ratio: float | None
    product_bbox: tuple[int, int, int, int] | None
    logo_bbox_in_frame: tuple[int, int, int, int] | None
    ocr_text: str | None
    ocr_available: bool
    failure_reasons: list[str]
    critic_name: str = "geometry"
    identity_score: int | None = None
    identity_verdict: str | None = None
    product_visible: bool | None = None
    product_count: int | None = None
    label_readability: str | None = None
    visual_drift: list[str] = field(default_factory=list)
    repair_instruction: str | None = None
    evidence: str | None = None
    temporal_consistency: str | None = None
    evidence_frames: list[str] = field(default_factory=list)
    raw_response: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
