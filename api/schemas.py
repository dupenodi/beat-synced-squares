"""Request/response models for the Beat-Synced Squares render API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobOptions(BaseModel):
    """Mirrors CLI flags accepted by `main.render_tracked_effect`."""

    model_config = ConfigDict(extra="forbid")

    fps: float | None = Field(default=None, description="Output FPS; omit for source rate")
    pts_per_beat: int = Field(default=20, ge=1, le=500)
    ambient_rate: float = Field(default=5.0, ge=0.0, le=200.0)
    jitter_px: float = Field(default=0.5, ge=0.0, le=20.0)
    life_frames: int = Field(default=10, ge=1, le=500)
    min_size: int = Field(default=15, ge=4, le=500)
    max_size: int = Field(default=40, ge=4, le=800)
    neighbor_links: int = Field(default=3, ge=0, le=20)
    orb_fast_threshold: int = Field(default=20, ge=1, le=255)
    bell_width: float = Field(default=4.0, ge=0.5, le=50.0)
    seed: int | None = Field(default=None)
    display_rotation: int | None = Field(
        default=None,
        description="Force display-rotation metadata in degrees; omit for ffprobe auto",
    )
    ignore_display_rotation: bool = Field(
        default=False,
        description="If true, do not rotate frames using container metadata",
    )

    @model_validator(mode="after")
    def _sizes(self) -> JobOptions:
        if self.max_size < self.min_size:
            raise ValueError("max_size must be greater than or equal to min_size")
        return self

    def to_render_kwargs(self) -> dict:
        d = self.model_dump()
        return {
            "fps": d["fps"],
            "pts_per_beat": d["pts_per_beat"],
            "ambient_rate": d["ambient_rate"],
            "jitter_px": d["jitter_px"],
            "life_frames": d["life_frames"],
            "min_size": d["min_size"],
            "max_size": d["max_size"],
            "neighbor_links": d["neighbor_links"],
            "orb_fast_threshold": d["orb_fast_threshold"],
            "bell_width": d["bell_width"],
            "seed": d["seed"],
            "display_rotation": d["display_rotation"],
            "ignore_display_rotation": d["ignore_display_rotation"],
        }


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    download_path: str | None = None
