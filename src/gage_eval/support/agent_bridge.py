from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .config import SupportConfig


@dataclass
class AgentResult:
    stdout: str
    stderr: str
    returncode: int


def parse_file_blocks(text: str) -> List[Tuple[Path, str]]:
    """Parse ### FILE: ... ### END blocks from agent output.

    All content outside protocol blocks is discarded.
    """

    blocks: List[Tuple[Path, List[str]]] = []
    current_path: Optional[Path] = None
    current_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("### FILE:"):
            # flush previous
            if current_path is not None:
                blocks.append((current_path, current_lines))
            path_str = line[len("### FILE:") :].strip()
            current_path = Path(path_str)
            current_lines = []
            continue
        if line == "### END":
            if current_path is not None:
                blocks.append((current_path, current_lines))
            current_path = None
            current_lines = []
            continue
        if current_path is not None:
            current_lines.append(raw)
    if current_path is not None:
        blocks.append((current_path, current_lines))
    return [(p, "\n".join(lines).rstrip() + "\n") for p, lines in blocks]


def call_agent(prompt: str, cfg: SupportConfig, *, prefer_stdin: bool = False) -> AgentResult:
    """Invoke configured agent as subprocess and return raw outputs."""

    raw_args = cfg.agent.build_yolo_args() if hasattr(cfg.agent, "build_yolo_args") else list(cfg.agent.yolo_args)
    command_name = Path(cfg.agent.command).name

    # OpenAI Codex CLI v2+: use `codex exec [OPTIONS] [PROMPT]`.
    if cfg.agent.type == "codex" and command_name == "codex":

        def _filter_codex_args(args: List[str]) -> List[str]:
            filtered: List[str] = []
            i = 0
            while i < len(args):
                a = args[i]
                if a == "exec":
                    logger.warning("agent.yolo_args 中包含 'exec'，已忽略（Support 会自动添加 codex exec）。")
                    i += 1
                    continue
                if a == "-y":
                    logger.warning("codex exec 不支持 -y，已自动忽略该参数。")
                    i += 1
                    continue
                if a == "--agent":
                    logger.warning("codex exec 不支持 --agent，已自动忽略该参数及其值。")
                    i += 2
                    continue
                if a in ("--reasoning-effort", "--reasoning_effort"):
                    logger.warning("codex exec 不支持 reasoning_effort，已自动忽略该参数及其值。")
                    i += 2
                    continue
                filtered.append(a)
                i += 1
            return filtered

        filtered_args = _filter_codex_args(raw_args)
        if "--skip-git-repo-check" not in filtered_args:
            filtered_args.append("--skip-git-repo-check")

        with tempfile.NamedTemporaryFile(prefix="gage_support_codex_", suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [cfg.agent.command, "exec"] + filtered_args + ["--output-last-message", str(tmp_path), "-"]
        logger.info(f"正在调用 Codex agent（timeout={cfg.agent.timeout}s），可能需要一些时间，请稍候…")
        logger.debug(f"Calling codex agent: {' '.join(cmd[:4])} ...")

        stop_event = threading.Event()

        def _heartbeat() -> None:
            start = time.monotonic()
            while not stop_event.wait(15):
                elapsed = int(time.monotonic() - start)
                logger.info(f"仍在等待 Codex 响应…已过去 {elapsed}s")

        threading.Thread(target=_heartbeat, daemon=True).start()
        start_ts = time.monotonic()

        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=cfg.agent.timeout,
                check=False,
            )
            last_msg = ""
            if tmp_path.exists():
                try:
                    last_msg = tmp_path.read_text(encoding="utf-8")
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            elapsed = time.monotonic() - start_ts
            logger.info(f"Codex 调用结束（耗时 {elapsed:.1f}s，returncode={completed.returncode}）。")
            return AgentResult(
                stdout=last_msg or (completed.stdout or ""),
                stderr=completed.stderr or "",
                returncode=completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            last_msg = ""
            if tmp_path.exists():
                try:
                    last_msg = tmp_path.read_text(encoding="utf-8")
                except Exception:
                    last_msg = ""
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            def _to_text(x: object) -> str:
                if x is None:
                    return ""
                if isinstance(x, bytes):
                    return x.decode("utf-8", errors="replace")
                return str(x)

            logger.error(
                f"Agent timeout after {cfg.agent.timeout}s. 模型响应超时，可尝试在 CLI 中使用 --timeout "
                "增大超时时间，或简化 Prompt 后重试。"
            )
            return AgentResult(
                stdout=last_msg or _to_text(getattr(exc, "stdout", "")),
                stderr=_to_text(getattr(exc, "stderr", "")) or "timeout",
                returncode=124,
            )
        except FileNotFoundError:
            logger.error(f"Agent command not found: {cfg.agent.command}")
            return AgentResult(stdout="", stderr="command not found", returncode=127)
        finally:
            stop_event.set()

    # Generic agent CLI: pass prompt via prompt_flag if provided.
    if cfg.agent.type == "gemini" and command_name == "gemini":
        cmd = [cfg.agent.command] + list(raw_args)

        logger.info(f"正在调用 Gemini agent（timeout={cfg.agent.timeout}s），可能需要一些时间，请稍候…")
        stop_event = threading.Event()

        def _heartbeat() -> None:
            start = time.monotonic()
            while not stop_event.wait(15):
                elapsed = int(time.monotonic() - start)
                logger.info(f"仍在等待 Agent 响应…已过去 {elapsed}s")

        threading.Thread(target=_heartbeat, daemon=True).start()

        run_kwargs = dict(
            capture_output=True,
            text=True,
            timeout=cfg.agent.timeout,
            check=False,
        )
        if prefer_stdin:
            run_kwargs["input"] = prompt
        elif cfg.agent.prompt_flag:
            cmd.extend([cfg.agent.prompt_flag, prompt])
        else:
            cmd.append(prompt)

        try:
            completed = subprocess.run(
                cmd,
                **run_kwargs,
            )
            return AgentResult(
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                returncode=completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error(
                f"Agent timeout after {cfg.agent.timeout}s. 模型响应超时，可尝试在 CLI 中使用 --timeout "
                "增大超时时间，或简化 Prompt 后重试。"
            )
            return AgentResult(stdout=exc.stdout or "", stderr=exc.stderr or "timeout", returncode=124)
        except FileNotFoundError:
            logger.error(f"Agent command not found: {cfg.agent.command}")
            return AgentResult(stdout="", stderr="command not found", returncode=127)
        finally:
            stop_event.set()

    cmd = [cfg.agent.command] + list(raw_args)
    run_kwargs_generic = dict(
        capture_output=True,
        text=True,
        timeout=cfg.agent.timeout,
        check=False,
    )
    if prefer_stdin:
        run_kwargs_generic["input"] = prompt
    elif cfg.agent.prompt_flag:
        cmd.extend([cfg.agent.prompt_flag, prompt])
    else:
        cmd.append(prompt)

    logger.debug(f"Calling agent: {' '.join(cmd[:3])} ...")

    stop_event = threading.Event()

    def _heartbeat() -> None:
        start = time.monotonic()
        while not stop_event.wait(15):
            elapsed = int(time.monotonic() - start)
            logger.info(f"仍在等待 Agent 响应…已过去 {elapsed}s")

    threading.Thread(target=_heartbeat, daemon=True).start()

    try:
        completed = subprocess.run(
            cmd,
            **run_kwargs_generic,
        )
        return AgentResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error(
            f"Agent timeout after {cfg.agent.timeout}s. 模型响应超时，可尝试在 CLI 中使用 --timeout "
            "增大超时时间，或简化 Prompt 后重试。"
        )
        return AgentResult(stdout=exc.stdout or "", stderr=exc.stderr or "timeout", returncode=124)
    except FileNotFoundError:
        logger.error(f"Agent command not found: {cfg.agent.command}")
        return AgentResult(stdout="", stderr="command not found", returncode=127)
    finally:
        stop_event.set()


def render_files_from_agent(prompt: str, cfg: SupportConfig) -> Dict[Path, str]:
    """Call agent and parse protocol blocks into files."""

    result = call_agent(prompt, cfg)
    if result.returncode != 0:
        raise RuntimeError(f"Agent failed (code={result.returncode}): {result.stderr.strip()}")
    blocks = parse_file_blocks(result.stdout)
    if not blocks and cfg.agent.type == "gemini":
        logger.warning("Agent output missing protocol blocks，使用 stdin 传递 Prompt 重试（Gemini）。")
        result = call_agent(prompt, cfg, prefer_stdin=True)
        if result.returncode != 0:
            raise RuntimeError(f"Agent failed (code={result.returncode}): {result.stderr.strip()}")
        blocks = parse_file_blocks(result.stdout)
    files: Dict[Path, str] = {path: content for path, content in blocks}
    if not files:
        preview = (result.stdout or "").strip().replace("\n", "\\n")
        if len(preview) > 800:
            preview = preview[:800] + "...<truncated>"
        raise RuntimeError(
            "Agent output did not contain any file protocol blocks. "
            "Expected '### FILE: <path>' ... '### END'. "
            f"Output preview: {preview}"
        )
    return files


def ping_agent(cfg: SupportConfig) -> bool:
    """Best-effort connectivity check by sending a simple prompt."""

    try:
        completed = call_agent("ping", cfg)
        if completed.returncode != 0:
            logger.error(f"Agent ping failed: {completed.stderr.strip()}")
            return False
        if completed.stdout.strip():
            logger.info(f"Agent ping response: {completed.stdout.strip()}")
        return True
    except Exception as exc:
        logger.error(f"Agent ping exception: {exc}")
        return False


__all__ = ["parse_file_blocks", "call_agent", "render_files_from_agent", "ping_agent", "AgentResult"]
