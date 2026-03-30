"""ArtifactLayout — frozen directory structure for trial artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolved output paths for a single runtime sample."""

    run_dir: str
    sample_dir: str
    agent_dir: str
    verifier_dir: str
    patch_file: str
    trajectory_file: str
    stdout_file: str
    metadata_file: str

    @classmethod
    def for_sample(cls, base_dir: str, run_id: str, sample_id: str) -> "ArtifactLayout":
        run_dir = os.path.join(base_dir, run_id)
        sample_dir = os.path.join(run_dir, "samples", sample_id)
        agent_dir = os.path.join(sample_dir, "agent")
        verifier_dir = os.path.join(sample_dir, "verifier")
        return cls(
            run_dir=run_dir,
            sample_dir=sample_dir,
            agent_dir=agent_dir,
            verifier_dir=verifier_dir,
            patch_file=os.path.join(agent_dir, "submission.patch"),
            trajectory_file=os.path.join(agent_dir, "trajectory.json"),
            stdout_file=os.path.join(agent_dir, "stdout.log"),
            metadata_file=os.path.join(sample_dir, "runtime_metadata.json"),
        )
