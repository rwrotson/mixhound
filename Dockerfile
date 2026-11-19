# syntax=docker/dockerfile:1
# Dependabot updates the pinned digest.
ARG PYTHON_IMAGE=python:3.14-slim@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2

FROM ${PYTHON_IMAGE} AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /usr/local/bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 PYTHONUNBUFFERED=1

# Keep dependency installation cached across source-only changes.
FROM base AS builder
# uv clones the deadwax git dependency.
RUN apt-get update && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM base AS development
ENV PATH="/app/.venv/bin:$PATH" MIXHOUND_SKILLS_DIR=/app/.agents/skills
# ffmpeg decodes preview clips for audio similarity.
# git lets uv clone the deadwax git dependency.
RUN apt-get update && apt-get install --yes --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY .agents ./.agents
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked
CMD ["uvicorn", "mixhound.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src", "--no-access-log"]

FROM ${PYTHON_IMAGE} AS production
LABEL org.opencontainers.image.description="mixhound crate-digging agent worker" \
      org.opencontainers.image.licenses="MIT"
WORKDIR /app
# ffmpeg decodes preview clips for audio similarity.
RUN apt-get update && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY .agents /app/.agents
# pip is unused at runtime; its bundled SBOM lists vendored libraries that scanners flag.
RUN python -m pip uninstall --yes --root-user-action=ignore pip \
    && useradd --system --uid 10001 --create-home --home-dir /home/app app \
    && mkdir -p /data && chown 10001 /data
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/home/app MIXHOUND_SKILLS_DIR=/app/.agents/skills MIXHOUND_DATA_DIR=/data
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=2)"]
CMD ["uvicorn", "mixhound.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--proxy-headers", "--timeout-graceful-shutdown", "20"]
