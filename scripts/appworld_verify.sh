#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APIS_URL="${APPWORLD_APIS_URL:-http://localhost:9000}"
MCP_URL="${APPWORLD_MCP_URL:-http://localhost:5001}"

appworld verify tasks --remote-apis-url "${APIS_URL}" --remote-mcp-url "${MCP_URL}"

if [[ -f "${ROOT_DIR}/scripts/call_mcp_server.py" ]]; then
  python "${ROOT_DIR}/scripts/call_mcp_server.py" --remote-mcp-url "${MCP_URL}" --remote-apis-url "${APIS_URL}"
fi
