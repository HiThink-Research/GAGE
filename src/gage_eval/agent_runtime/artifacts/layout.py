"""Frozen directory structure for trial artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactLayout:
    """Filesystem layout for a single sample run."""

    run_dir: str
    task_dir: str
    sample_dir: str
    canonical_sample_dir: str
    sample_file: str
    agent_dir: str
    verifier_dir: str
    patch_file: str
    trajectory_file: str
    stdout_file: str
    stderr_file: str
    final_message_file: str
    metadata_file: str
    verifier_result_file: str
    verifier_stdout_file: str
    verifier_stderr_file: str
    verifier_logs_dir: str
    verifier_workspace_dir: str
    attachments_dir: str

    @classmethod
    def for_sample(
        cls,
        base_dir: str,
        run_id: str,
        sample_id: str,
        *,
        task_id: str | None = None,
    ) -> "ArtifactLayout":
        """Build the default layout for a sample."""
        run_dir = os.path.join(base_dir, run_id)
        task_dir = os.path.join(run_dir, "samples", _sanitize_path_part(task_id or "global"))
        sample_dir = os.path.join(task_dir, _sanitize_path_part(sample_id))
        canonical_sample_dir = sample_dir
        agent_dir = os.path.join(sample_dir, "agent")
        verifier_dir = os.path.join(sample_dir, "verifier")
        return cls(
            run_dir=run_dir,
            task_dir=task_dir,
            sample_dir=sample_dir,
            canonical_sample_dir=canonical_sample_dir,
            sample_file=os.path.join(sample_dir, "sample.json"),
            agent_dir=agent_dir,
            verifier_dir=verifier_dir,
            patch_file=os.path.join(agent_dir, "submission.patch"),
            trajectory_file=os.path.join(agent_dir, "trajectory.json"),
            stdout_file=os.path.join(agent_dir, "stdout.log"),
            stderr_file=os.path.join(agent_dir, "stderr.log"),
            final_message_file=os.path.join(agent_dir, "final_message.md"),
            metadata_file=os.path.join(sample_dir, "runtime_metadata.json"),
            verifier_result_file=os.path.join(verifier_dir, "result.json"),
            verifier_stdout_file=os.path.join(verifier_dir, "stdout.log"),
            verifier_stderr_file=os.path.join(verifier_dir, "stderr.log"),
            verifier_logs_dir=os.path.join(verifier_dir, "logs"),
            verifier_workspace_dir=os.path.join(verifier_dir, "workspace"),
            attachments_dir=os.path.join(sample_dir, "attachments"),
        )


def _sanitize_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
