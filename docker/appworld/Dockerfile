ARG APPWORLD_PLATFORM=linux/amd64
FROM --platform=$APPWORLD_PLATFORM ghcr.io/stonybrooknlp/appworld:latest

WORKDIR /run
ENV VIRTUAL_ENV=
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

# Install build dependencies (gcc, python3-dev, etc.)
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install AppWorld (source preferred to ensure MCP + serve multiple)
ARG APPWORLD_VERSION=source
ARG APPWORLD_SOURCE_DIR=reference/agents/appworld-main
COPY ${APPWORLD_SOURCE_DIR} /tmp/appworld-src
RUN if [ "$APPWORLD_VERSION" = "source" ]; then \
        if [ ! -d "/tmp/appworld-src" ]; then \
            echo "AppWorld source not found at ${APPWORLD_SOURCE_DIR}. Update APPWORLD_SOURCE_DIR or build from repo root."; \
            exit 1; \
        fi; \
        pip install --upgrade "/tmp/appworld-src[mcp]"; \
    elif [ "$APPWORLD_VERSION" = "latest" ]; then \
        pip install --upgrade "appworld[mcp]"; \
    else \
        pip install "appworld[mcp]==$APPWORLD_VERSION"; \
    fi && \
    pip install --upgrade "typer==0.16.0" "click==8.1.7" && \
    rm -rf /tmp/appworld-src

# Ensure AppWorld bundles are real files (not Git LFS pointers).
RUN python - <<'PY'
import os
import sysconfig


def is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as file:
            head = file.read(256)
    except FileNotFoundError:
        return True
    return b"git-lfs" in head or b"version https://git-lfs.github.com/spec/v1" in head


root = os.path.join(sysconfig.get_paths()["purelib"], "appworld", ".source")
missing = []
for name in ("apps.bundle", "tests.bundle"):
    path = os.path.join(root, name)
    if is_lfs_pointer(path):
        missing.append(path)
if missing:
    raise SystemExit(
        "AppWorld bundles are Git LFS pointers. Run `git lfs pull` in "
        "reference/agents/appworld-main before building, or set APPWORLD_VERSION=latest."
    )
print(f"AppWorld bundles ready: {root}")
PY

# Prepare AppWorld data during build for offline runtime
ARG APPWORLD_DOWNLOAD_DATA=1
ARG APPWORLD_DATA_MODE=minimal
ENV APPWORLD_ROOT=/run
RUN /usr/local/bin/appworld install && \
    if [ "$APPWORLD_DOWNLOAD_DATA" = "1" ]; then \
        if /usr/local/bin/appworld download data --help | grep -q -- "--mode"; then \
            /usr/local/bin/appworld download data --mode "$APPWORLD_DATA_MODE"; \
        else \
            /usr/local/bin/appworld download data; \
        fi; \
    fi

# Expose ports
EXPOSE 8000 9000 5001

# Start official servers
ENTRYPOINT ["/usr/local/bin/appworld", "serve"]
CMD ["multiple", "--environment", "--port 8000", "--apis", "--port 9000", "--mcp", "http --port 5001 --output-type structured_data_only", "--root", "/run"]
