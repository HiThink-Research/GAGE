"""Probe AppWorld tool docs size for Meta-Tool planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from gage_eval.mcp import McpClient
from gage_eval.role.adapters.tool_docs import build_app_catalog, build_tool_documentation


def load_tools_from_file(path: Path) -> List[Dict[str, Any]]:
    """Load MCP tool definitions from a JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        tools = payload.get("tools")
        if isinstance(tools, list):
            return tools
    raise ValueError("tools file must be a list or include a 'tools' list")


def fetch_tools_from_mcp(endpoint: str, transport: str, timeout_s: int) -> List[Dict[str, Any]]:
    """Fetch MCP tool definitions from a live endpoint."""

    client = McpClient(
        mcp_client_id="appworld_probe",
        transport=transport,
        endpoint=endpoint,
        timeout_s=timeout_s,
    )
    return list(client.list_tools())


def build_probe_report(
    tools: Sequence[Dict[str, Any]],
    *,
    allowed_apps: Sequence[str] | None = None,
    doc_format: str = "text",
    max_endpoints: int | None = None,
    max_chars: int | None = None,
) -> Dict[str, Any]:
    """Generate a documentation probe report."""

    catalog = build_app_catalog(tools, allowed_apps=allowed_apps)
    documentation = build_tool_documentation(
        catalog,
        doc_format=doc_format,
        max_endpoints=max_endpoints,
        max_chars=max_chars,
    )
    app_counts = {app: len(endpoints) for app, endpoints in documentation.endpoints_by_app.items()}
    report = dict(documentation.meta)
    report["apps_detail"] = app_counts
    report["doc_preview"] = documentation.text
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe AppWorld tool documentation size.")
    parser.add_argument("--tools-file", type=Path, help="Path to a JSON file containing MCP tools.")
    parser.add_argument("--mcp-endpoint", type=str, help="MCP endpoint URL to list tools from.")
    parser.add_argument("--mcp-transport", type=str, default="streamable_http", help="MCP transport type.")
    parser.add_argument("--timeout-s", type=int, default=30, help="MCP request timeout in seconds.")
    parser.add_argument("--allowed-apps", nargs="*", default=None, help="Optional app allowlist.")
    parser.add_argument("--doc-format", type=str, default="text", help="Documentation format.")
    parser.add_argument("--max-endpoints", type=int, default=None, help="Max endpoints per app.")
    parser.add_argument("--max-chars", type=int, default=None, help="Max characters in documentation.")
    parser.add_argument("--print-doc", action="store_true", help="Print documentation preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.tools_file) == bool(args.mcp_endpoint):
        raise SystemExit("Provide exactly one of --tools-file or --mcp-endpoint.")

    if args.tools_file:
        tools = load_tools_from_file(args.tools_file)
    else:
        tools = fetch_tools_from_mcp(args.mcp_endpoint, args.mcp_transport, args.timeout_s)

    report = build_probe_report(
        tools,
        allowed_apps=args.allowed_apps,
        doc_format=args.doc_format,
        max_endpoints=args.max_endpoints,
        max_chars=args.max_chars,
    )

    summary = json.dumps({k: v for k, v in report.items() if k != "doc_preview"}, indent=2, ensure_ascii=True)
    print(summary)
    if args.print_doc:
        print("\n--- tool_documentation ---\n")
        print(report["doc_preview"])


__all__ = [
    "build_probe_report",
    "fetch_tools_from_mcp",
    "load_tools_from_file",
    "main",
    "parse_args",
]
