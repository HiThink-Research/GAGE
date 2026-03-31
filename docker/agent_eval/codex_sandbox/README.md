# Codex Sandbox

This image is the project-owned Docker sandbox for installed-agent evaluation.

## Build

```bash
cd /Users/shuo/code/GAGE_workspace/repo
docker build -t gage-codex-sandbox:latest docker/agent_eval/codex_sandbox
```

## Runtime Contract

- workdir: `/workspace`
- Codex home: `/agent`
- expected env:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL` (optional)

The acceptance configs mount the repository root into `/workspace` and run the
installed client inside the container.

## Acceptance Configs

- `config/custom/acceptance/docker_installed_client_swebench.yaml`
- `config/custom/acceptance/docker_installed_client_terminal_bench.yaml`
