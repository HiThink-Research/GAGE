"""Utilities for SWE-bench Pro integrations."""

from __future__ import annotations

import platform as py_platform
from typing import Optional


def get_dockerhub_image_uri(uid: str, dockerhub_username: str, repo_name: str = "") -> str:
    """Return the Docker Hub image URI following SWE-bench Pro conventions."""

    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"{dockerhub_username}/sweap-images:{tag}"


def resolve_docker_platform(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve docker platform override (defaults to linux/amd64 on arm64 hosts)."""

    if explicit:
        return str(explicit)
    try:
        machine = py_platform.machine().lower()
    except Exception:
        return None
    if machine in {"arm64", "aarch64"}:
        return "linux/amd64"
    return None
