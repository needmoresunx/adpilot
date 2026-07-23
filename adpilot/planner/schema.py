from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ShotPlan:
    shot_id: str
    duration: int
    goal: str
    background_prompt: str
    product_position: str
    product_scale: float
    motion_prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdPlan:
    concept: str
    style: str
    shots: list[ShotPlan]

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "style": self.style,
            "shots": [shot.to_dict() for shot in self.shots],
        }
