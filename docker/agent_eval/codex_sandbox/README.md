# Codex Sandbox

This image is the project-owned Docker sandbox for installed-agent evaluation.

## Build

```bash
cd /Users/shuo/code/GAGE_workspace/repo
docker build -t gage-codex-sandbox:latest docker/agent_eval/codex_sandbox
```

## Runtime Contract

- workdir: `/workspace`
- Codex home: `/agent` unless overridden by `CODEX_HOME`
- expected env:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL` (optional)
  - `GAGE_CODEX_HOST_HOME` (optional read-only host auth mount such as `/gage-host-codex`)

If `GAGE_CODEX_HOST_HOME` is mounted into the container and `auth.json` is absent,
the bootstrap script copies the host Codex login state into `CODEX_HOME` before
executing the client. This allows Docker-based installed-client runs to reuse a
host `codex login` session without exporting an API key.

The acceptance configs mount the repository root into `/workspace` and run the
installed client inside the container.

## Acceptance Configs

- `config/custom/acceptance/docker_installed_client_swebench.yaml`
- `config/custom/acceptance/docker_installed_client_terminal_bench.yaml`
