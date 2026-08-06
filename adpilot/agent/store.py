from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODELS = {
    "vlm": "Qwen/Qwen2.5-VL-3B-Instruct",
    "keyframe": "black-forest-labs/FLUX.1-Kontext-dev",
    "video": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(path: Path) -> str:
    suffix = path.suffix.lower()
    stem = "".join(char if char.isalnum() or char in "-_" else "-" for char in path.stem).strip("-")
    return f"{stem or 'asset'}{suffix}"


def _project_id(root: Path, product_path: Path, project_name: str | None) -> str:
    label_source = project_name or product_path.stem
    label = "".join(char.lower() if char.isalnum() else "-" for char in label_source).strip("-")
    label = label[:48].strip("-") or "product"
    date_prefix = datetime.now().strftime("%Y%m%d")
    candidate = f"{date_prefix}_{label}"
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{date_prefix}_{label}-{suffix:02d}"
        suffix += 1
    return candidate


@dataclass
class ProjectConfig:
    project_id: str
    brand: str
    prompt: str
    mode: str
    product_asset_id: str
    reference_asset_ids: list[str]
    product_category: str | None = None
    product_description: str | None = None
    target_audience: str | None = None
    ad_mood: str | None = None
    identity_anchors: str | None = None
    package_state: str | None = None
    platform: str = "landscape"
    duration: int = 9
    max_agent_steps: int = 16
    max_keyframe_attempts: int = 2
    max_video_attempts: int = 2
    keyframe_candidates: int = 2
    video_candidates: int = 1
    keyframe_seed: int = 17
    video_seed: int = 11
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectConfig":
        merged_models = dict(DEFAULT_MODELS)
        merged_models.update(value.get("models") or {})
        return cls(
            project_id=str(value["project_id"]),
            brand=str(value["brand"]),
            prompt=str(value["prompt"]),
            mode=str(value["mode"]),
            product_asset_id=str(value["product_asset_id"]),
            reference_asset_ids=[str(item) for item in value.get("reference_asset_ids", [])],
            product_category=value.get("product_category"),
            product_description=value.get("product_description"),
            target_audience=value.get("target_audience"),
            ad_mood=value.get("ad_mood"),
            identity_anchors=value.get("identity_anchors"),
            package_state=value.get("package_state"),
            platform=str(value.get("platform", "landscape")),
            duration=int(value.get("duration", 9)),
            max_agent_steps=int(value.get("max_agent_steps", 16)),
            max_keyframe_attempts=int(value.get("max_keyframe_attempts", 2)),
            max_video_attempts=int(value.get("max_video_attempts", 2)),
            keyframe_candidates=int(value.get("keyframe_candidates", 2)),
            video_candidates=int(value.get("video_candidates", 1)),
            keyframe_seed=int(value.get("keyframe_seed", 17)),
            video_seed=int(value.get("video_seed", 11)),
            models=merged_models,
        )


class ProjectStore:
    """Project-local asset metadata and event history.

    LangGraph owns checkpointed agent state. This store owns user-visible assets,
    revision folders, and audit events so all file provenance remains explicit.
    """

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.database_path = self.project_dir / "project.db"
        self.config_path = self.project_dir / "project.json"
        if not self.config_path.is_file():
            raise FileNotFoundError(f"AdPilot project metadata not found: {self.config_path}")
        self._initialize_database()

    @classmethod
    def create(
        cls,
        root: Path,
        product_path: Path,
        brand: str,
        prompt: str,
        mode: str,
        reference_paths: list[Path] | None = None,
        project_name: str | None = None,
        **settings: Any,
    ) -> "ProjectStore":
        if mode not in {"auto", "guided"}:
            raise ValueError("mode must be 'auto' or 'guided'")
        if not brand.strip() or not prompt.strip():
            raise ValueError("brand and prompt are required")
        product_path = Path(product_path)
        if not product_path.is_file():
            raise FileNotFoundError(f"Product image not found: {product_path}")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        project_id = _project_id(root, product_path, project_name)
        project_dir = root / project_id
        project_dir.mkdir(parents=True, exist_ok=False)
        (project_dir / "assets").mkdir()
        (project_dir / "revisions").mkdir()
        store = cls._open_new(project_dir)
        product_asset_id = store.add_asset(product_path, role="product_reference", origin="uploaded")
        reference_asset_ids = [product_asset_id]
        for path in reference_paths or []:
            reference_path = Path(path)
            if not reference_path.is_file():
                raise FileNotFoundError(f"Reference image not found: {reference_path}")
            reference_asset_ids.append(store.add_asset(reference_path, role="product_reference", origin="uploaded"))
        config = ProjectConfig(
            project_id=project_id,
            brand=brand.strip(),
            prompt=prompt.strip(),
            mode=mode,
            product_asset_id=product_asset_id,
            reference_asset_ids=reference_asset_ids,
            **settings,
        )
        store.config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        store.record_event("project_created", {"mode": mode, "asset_count": len(reference_asset_ids)})
        return store

    @classmethod
    def _open_new(cls, project_dir: Path) -> "ProjectStore":
        database_path = project_dir / "project.db"
        connection = sqlite3.connect(database_path)
        connection.close()
        placeholder = project_dir / "project.json"
        placeholder.write_text("{}", encoding="utf-8")
        return cls(project_dir)

    @property
    def config(self) -> ProjectConfig:
        return ProjectConfig.from_dict(json.loads(self.config_path.read_text(encoding="utf-8")))

    def revision_dir(self, revision: int) -> Path:
        directory = self.project_dir / "revisions" / f"rev_{revision:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    parent_asset_ids TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    revision INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )

    def add_asset(
        self,
        source_path: Path,
        role: str,
        origin: str,
        revision: int | None = None,
        parent_asset_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Asset file not found: {source_path}")
        asset_id = f"asset_{uuid.uuid4().hex[:12]}"
        destination = self.project_dir / "assets" / f"{asset_id}_{_safe_name(source_path)}"
        shutil.copy2(source_path, destination)
        relative_path = destination.relative_to(self.project_dir).as_posix()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO assets(asset_id, role, origin, relative_path, parent_asset_ids, metadata, revision, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    role,
                    origin,
                    relative_path,
                    json.dumps(parent_asset_ids or []),
                    json.dumps(metadata or {}),
                    revision,
                    utc_now(),
                ),
            )
        return asset_id

    def register_generated_asset(
        self,
        path: Path,
        role: str,
        revision: int,
        parent_asset_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        path = Path(path)
        try:
            relative_path = path.resolve().relative_to(self.project_dir.resolve()).as_posix()
        except ValueError:
            return self.add_asset(
                path,
                role=role,
                origin="generated",
                revision=revision,
                parent_asset_ids=parent_asset_ids,
                metadata=metadata,
            )
        asset_id = f"asset_{uuid.uuid4().hex[:12]}"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO assets(asset_id, role, origin, relative_path, parent_asset_ids, metadata, revision, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    role,
                    "generated",
                    relative_path,
                    json.dumps(parent_asset_ids or []),
                    json.dumps(metadata or {}),
                    revision,
                    utc_now(),
                ),
            )
        return asset_id

    def asset_path(self, asset_id: str) -> Path:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT relative_path FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown asset id: {asset_id}")
        return self.project_dir / str(row["relative_path"])

    def asset_records(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM assets ORDER BY created_at").fetchall()
        return [
            {
                **dict(row),
                "parent_asset_ids": json.loads(row["parent_asset_ids"]),
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO events(event_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (f"event_{uuid.uuid4().hex[:12]}", event_type, json.dumps(payload or {}), utc_now()),
            )

    def events(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY created_at").fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]
