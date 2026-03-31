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
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @openai/codex@latest \
    && codex --version

RUN mkdir -p /workspace /agent

COPY scripts/bootstrap_codex.sh /usr/local/bin/bootstrap_codex.sh
RUN chmod +x /usr/local/bin/bootstrap_codex.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/bootstrap_codex.sh"]
CMD ["sleep", "infinity"]
