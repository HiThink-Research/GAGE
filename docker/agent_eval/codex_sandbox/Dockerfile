FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV CODEX_HOME=/agent
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Minimal Python deps required for mounted GAGE smoke scripts.
RUN pip3 install --break-system-packages --no-cache-dir \
        loguru \
        requests

RUN npm install -g @openai/codex@latest \
    && codex --version

RUN mkdir -p /workspace /agent

COPY docker/agent_eval/codex_sandbox/scripts/bootstrap_codex.sh /usr/local/bin/bootstrap_codex.sh
RUN chmod +x /usr/local/bin/bootstrap_codex.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/bootstrap_codex.sh"]
CMD ["sleep", "infinity"]
