"""SWE-bench Pro repository context provider."""

from __future__ import annotations

import ast
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from gage_eval.registry import registry


_DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
}


@registry.asset(
    "context_impls",
    "swebench_repo",
    desc="SWE-bench Pro repo context provider",
    tags=("swebench", "context"),
)
class SwebenchRepoContext:
    def __init__(
        self,
        *,
        repo_source: str = "docker_image",
        repo_root: str = "/app",
        topk_files: int = 20,
        repo_map_max_files: int = 0,
        repo_map_max_lines: int = 120,
        repo_map_scan_lines: int = 400,
        repo_map_max_symbols_per_file: int = 4,
        max_tree_depth: int = 3,
        max_tree_lines: int = 200,
        max_file_lines: int = 200,
        max_file_chars: int = 8000,
        dockerhub_username: str = "jefzda",
        registry_prefix: Optional[str] = None,
        docker_platform: Optional[str] = None,
        exclude_dirs: Optional[Sequence[str]] = None,
        block_network: bool = True,
        exec_timeout_s: int = 30,
    ) -> None:
        self._repo_source = repo_source
        self._repo_root = repo_root
        self._topk_files = max(0, int(topk_files))
        self._repo_map_max_files = max(0, int(repo_map_max_files))
        self._repo_map_max_lines = max(1, int(repo_map_max_lines))
        self._repo_map_scan_lines = max(1, int(repo_map_scan_lines))
        self._repo_map_max_symbols_per_file = max(1, int(repo_map_max_symbols_per_file))
        self._max_tree_depth = max(1, int(max_tree_depth))
        self._max_tree_lines = max(1, int(max_tree_lines))
        self._max_file_lines = max(1, int(max_file_lines))
        self._max_file_chars = max(100, int(max_file_chars))
        self._dockerhub_username = dockerhub_username
        self._registry_prefix = registry_prefix
        self._docker_platform = docker_platform
        self._exclude_dirs = set(exclude_dirs or _DEFAULT_EXCLUDE_DIRS)
        self._block_network = bool(block_network)
        self._exec_timeout_s = max(1, int(exec_timeout_s))

    def provide(self, payload: Dict[str, Any], _state=None) -> Dict[str, Any]:
        sample = payload.get("sample") or {}
        params = payload.get("params") or {}
        repo_source = params.get("repo_source", self._repo_source)
        repo_root = params.get("repo_root", self._repo_root)
        topk_files = int(params.get("topk_files", self._topk_files))
        repo_map_max_files = int(params.get("repo_map_max_files", self._repo_map_max_files))
        repo_map_max_lines = int(params.get("repo_map_max_lines", self._repo_map_max_lines))
        repo_map_scan_lines = int(params.get("repo_map_scan_lines", self._repo_map_scan_lines))
        repo_map_max_symbols_per_file = int(
            params.get("repo_map_max_symbols_per_file", self._repo_map_max_symbols_per_file)
        )
        max_tree_depth = int(params.get("max_tree_depth", self._max_tree_depth))
        max_tree_lines = int(params.get("max_tree_lines", self._max_tree_lines))
        max_file_lines = int(params.get("max_file_lines", self._max_file_lines))
        max_file_chars = int(params.get("max_file_chars", self._max_file_chars))
        exclude_dirs = set(params.get("exclude_dirs", self._exclude_dirs))
        exec_timeout_s = int(params.get("exec_timeout_s", self._exec_timeout_s))

        logger.debug("SWE-bench repo context repo_source={} repo_root={}", repo_source, repo_root)

        if repo_source in {"local", "local_path"}:
            repo_path = _resolve_local_repo(sample, params)
            tree_text = _build_tree_local(repo_path, max_tree_depth, max_tree_lines, exclude_dirs)
            all_files = list(_list_local_files(repo_path, exclude_dirs))
            file_reader = _LocalFileReader(repo_path)
        elif repo_source == "docker_image":
            sandbox_provider = payload.get("sandbox_provider")
            if sandbox_provider is None:
                raise ValueError("swebench_repo requires sandbox_provider for repo_source=docker_image")
            handle = sandbox_provider.get_handle()
            if handle is None or handle.sandbox is None:
                raise ValueError("swebench_repo sandbox handle missing for repo_source=docker_image")
            session = _SandboxRepoSession(handle.sandbox, repo_root=repo_root, exec_timeout_s=exec_timeout_s)
            tree_text = session.list_tree(max_tree_depth, max_tree_lines)
            all_files = session.list_files(exclude_dirs)
            file_reader = session
            repo_map = _build_repo_map(
                file_reader,
                sample,
                all_files,
                max_files=repo_map_max_files,
                max_lines=repo_map_max_lines,
                scan_lines=repo_map_scan_lines,
                max_symbols_per_file=repo_map_max_symbols_per_file,
            )
            return _inject_context(
                sample,
                tree_text,
                _collect_file_snippets(
                    file_reader,
                    _select_files(sample, all_files, topk_files),
                    max_file_lines,
                    max_file_chars,
                ),
                repo_map,
            )
        else:
            raise ValueError(f"Unsupported repo_source: {repo_source}")

        file_snippets = _collect_file_snippets(
            file_reader,
            _select_files(sample, all_files, topk_files),
            max_file_lines,
            max_file_chars,
        )
        repo_map = _build_repo_map(
            file_reader,
            sample,
            all_files,
            max_files=repo_map_max_files,
            max_lines=repo_map_max_lines,
            scan_lines=repo_map_scan_lines,
            max_symbols_per_file=repo_map_max_symbols_per_file,
        )
        return _inject_context(sample, tree_text, file_snippets, repo_map)


class _LocalFileReader:
    def __init__(self, repo_path: Path) -> None:
        self._repo_path = repo_path

    def read_file(self, rel_path: str, max_lines: int) -> str:
        target = self._repo_path / rel_path
        try:
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                lines = []
                for _ in range(max_lines):
                    line = handle.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines)
        except FileNotFoundError:
            return ""


class _SandboxRepoSession:
    def __init__(self, sandbox, repo_root: str, exec_timeout_s: int) -> None:
        self._sandbox = sandbox
        self._repo_root = repo_root
        self._exec_timeout_s = max(1, int(exec_timeout_s))

    def list_tree(self, max_depth: int, max_lines: int) -> str:
        exit_code, output = self._exec(f"find . -maxdepth {max_depth} -print")
        if exit_code != 0:
            return ""
        lines = [ln for ln in output.splitlines() if ln.strip()]
        return "\n".join(lines[:max_lines])

    def list_files(self, exclude_dirs: Sequence[str]) -> List[str]:
        exit_code, output = self._exec("find . -type f -print")
        if exit_code != 0:
            return []
        cleaned = []
        for line in output.splitlines():
            rel = line.strip()
            if rel.startswith("./"):
                rel = rel[2:]
            if not rel:
                continue
            if _is_excluded(rel, exclude_dirs):
                continue
            cleaned.append(rel)
        return cleaned

    def read_file(self, rel_path: str, max_lines: int) -> str:
        rel = rel_path.lstrip("./")
        cmd = f"sed -n '1,{max_lines}p' {shlex.quote(rel)}"
        exit_code, output = self._exec(cmd)
        return output if exit_code == 0 else ""

    def _exec(self, cmd: str) -> Tuple[int, str]:
        full_cmd = f"cd {shlex.quote(self._repo_root)} && {cmd}"
        result = self._sandbox.exec(full_cmd, timeout=self._exec_timeout_s)
        output = result.stdout if isinstance(result.stdout, str) else str(result.stdout or "")
        return int(result.exit_code or 0), output


def _resolve_local_repo(sample: Dict[str, Any], params: Dict[str, Any]) -> Path:
    for key in ("repo_path", "local_repo_root", "local_path"):
        value = params.get(key)
        if value:
            return Path(value).expanduser().resolve()
    metadata = sample.get("metadata") or {}
    for key in ("repo_path", "local_repo_root", "local_path"):
        value = metadata.get(key)
        if value:
            return Path(value).expanduser().resolve()
    raise ValueError("local repo path not provided (repo_source=local_path)")




def _build_tree_local(
    repo_path: Path,
    max_depth: int,
    max_lines: int,
    exclude_dirs: Sequence[str],
) -> str:
    lines: List[str] = []
    for root, dirs, files in os.walk(repo_path):
        rel_root = Path(root).relative_to(repo_path)
        depth = len(rel_root.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        rel_str = "." if not rel_root.parts else "./" + "/".join(rel_root.parts)
        lines.append(rel_str + "/")
        for name in sorted(files):
            if len(lines) >= max_lines:
                break
            rel_file = rel_root / name if rel_root.parts else Path(name)
            if _is_excluded(str(rel_file), exclude_dirs):
                continue
            if depth + 1 <= max_depth:
                lines.append("./" + str(rel_file))
        if len(lines) >= max_lines:
            break
    return "\n".join(lines[:max_lines])


def _list_local_files(repo_path: Path, exclude_dirs: Sequence[str]) -> Iterable[str]:
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        rel_root = Path(root).relative_to(repo_path)
        for name in files:
            rel_file = rel_root / name if rel_root.parts else Path(name)
            rel_str = str(rel_file)
            if _is_excluded(rel_str, exclude_dirs):
                continue
            yield rel_str


def _select_files(sample: Dict[str, Any], all_files: Sequence[str], topk: int) -> List[str]:
    if topk <= 0:
        return []
    candidates = _extract_test_paths(sample)
    selected: List[str] = []
    normalized_all = [f.lstrip("./") for f in all_files]
    all_lower = [f.lower() for f in normalized_all]
    for cand in candidates:
        if cand in selected:
            continue
        if cand in normalized_all:
            selected.append(cand)
            if len(selected) >= topk:
                return selected
    query = _resolve_query(sample)
    keywords = _extract_keywords(query)
    if keywords:
        for idx, rel in enumerate(normalized_all):
            if rel in selected:
                continue
            if any(k in all_lower[idx] for k in keywords):
                selected.append(rel)
                if len(selected) >= topk:
                    return selected
    for rel in normalized_all:
        if rel in selected:
            continue
        selected.append(rel)
        if len(selected) >= topk:
            break
    return selected


def _collect_file_snippets(
    reader,
    paths: Sequence[str],
    max_lines: int,
    max_chars: int,
) -> Dict[str, str]:
    snippets: Dict[str, str] = {}
    for rel_path in paths:
        content = reader.read_file(rel_path, max_lines)
        if not content:
            continue
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [truncated]"
        snippets[rel_path] = content
    return snippets


_REPO_MAP_PATTERNS = [
    re.compile(r"^\s*(class|interface|enum)\s+\w+"),
    re.compile(r"^\s*(export\s+)?(class|function)\s+\w+"),
    re.compile(r"^\s*(async\s+)?def\s+\w+"),
    re.compile(r"^\s*func\s+\w+"),
    re.compile(r"^\s*func\s*\([^)]*\)\s*\w+"),
    re.compile(r"^\s*type\s+\w+\s+struct\b"),
]


def _build_repo_map(
    reader,
    sample: Dict[str, Any],
    all_files: Sequence[str],
    *,
    max_files: int,
    max_lines: int,
    scan_lines: int,
    max_symbols_per_file: int,
) -> str:
    if max_files <= 0 or max_lines <= 0:
        return ""
    selected = _select_files(sample, all_files, max_files)
    lines: List[str] = []
    line_count = 0
    for rel_path in selected:
        if line_count >= max_lines:
            break
        content = reader.read_file(rel_path, scan_lines)
        symbols = _extract_symbols(content, max_symbols_per_file)
        if not symbols:
            continue
        if line_count + 1 > max_lines:
            break
        lines.append(f"{rel_path}:")
        line_count += 1
        for symbol in symbols:
            if line_count >= max_lines:
                break
            lines.append(f"  - {symbol}")
            line_count += 1
    return "\n".join(lines)


def _extract_symbols(content: str, max_symbols: int) -> List[str]:
    symbols: List[str] = []
    if not content:
        return symbols
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or "\x00" in stripped:
            continue
        if any(pattern.match(stripped) for pattern in _REPO_MAP_PATTERNS):
            symbols.append(stripped)
            if len(symbols) >= max_symbols:
                break
    return symbols


def _inject_context(
    sample: Dict[str, Any],
    tree_text: str,
    file_snippets: Dict[str, str],
    repo_map: str,
) -> Dict[str, Any]:
    context_blocks = []
    if tree_text:
        context_blocks.append("Repository Tree:\n" + tree_text)
    if file_snippets:
        rendered = []
        for rel_path, content in file_snippets.items():
            rendered.append(f"### {rel_path}\n```\n{content}\n```")
        context_blocks.append("Relevant Files:\n" + "\n\n".join(rendered))
    if repo_map:
        context_blocks.append("Repo Map:\n" + repo_map)
    context_text = "\n\n".join(context_blocks)
    if context_text:
        _append_prompt(sample, context_text)
    return {
        "repo_tree": tree_text,
        "selected_files": list(file_snippets.keys()),
        "repo_map": repo_map,
    }


def _append_prompt(sample: Dict[str, Any], context_text: str) -> None:
    inputs = sample.get("inputs")
    if isinstance(inputs, dict):
        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and prompt:
            inputs["prompt"] = prompt + "\n\n" + context_text
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = content + "\n\n" + context_text
        elif isinstance(content, list):
            content.append({"type": "text", "text": context_text})
        elif content is None:
            last["content"] = context_text


def _resolve_query(sample: Dict[str, Any]) -> str:
    inputs = sample.get("inputs") or {}
    if isinstance(inputs, dict):
        prompt = inputs.get("prompt")
        if isinstance(prompt, str):
            return prompt
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        content = messages[-1].get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [frag.get("text") for frag in content if isinstance(frag, dict)]
            return "\n".join([t for t in texts if t])
    return ""


def _extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9_./-]{3,}", text.lower())
    seen = set()
    keywords: List[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)
        if len(keywords) >= 30:
            break
    return keywords


def _extract_test_paths(sample: Dict[str, Any]) -> List[str]:
    metadata = sample.get("metadata") or {}
    values = []
    for key in ("selected_test_files_to_run", "fail_to_pass", "pass_to_pass"):
        if key in metadata:
            values.append(metadata.get(key))
    paths: List[str] = []
    for value in values:
        for entry in _as_list(value):
            path = _extract_path_from_test(entry)
            if not path:
                continue
            if path.startswith("/app/"):
                path = path[len("/app/") :]
            if path.startswith("./"):
                path = path[2:]
            if path not in paths:
                paths.append(path)
    return paths


def _extract_path_from_test(entry: str) -> str:
    if "::" in entry:
        return entry.split("::", 1)[0].strip()
    if " | " in entry:
        return entry.split(" | ", 1)[0].strip()
    return entry.strip()


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        if "\n" in raw:
            return [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if "," in raw:
            return [seg.strip() for seg in raw.split(",") if seg.strip()]
        return [raw]
    return [str(value)]


def _is_excluded(rel_path: str, exclude_dirs: Sequence[str]) -> bool:
    parts = Path(rel_path).parts
    for part in parts[:-1]:
        if part in exclude_dirs:
            return True
    return False
