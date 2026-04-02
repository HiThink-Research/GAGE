"""Compatibility re-export for formal client surface definitions."""

from __future__ import annotations

from gage_eval.sandbox.surfaces import (
    ClientSurface,
    SurfaceStatus,
    SurfaceType,
    build_remote_surfaces,
    serialize_surfaces,
)

__all__ = [
    "ClientSurface",
    "SurfaceStatus",
    "SurfaceType",
    "build_remote_surfaces",
    "serialize_surfaces",
]
