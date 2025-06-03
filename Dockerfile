# Stage 1: Build dependencies using uv
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Set uv environment variables for optimized linking and bytecode compilation
ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy

# Install dependencies into a virtual environment using uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-editable --no-install-project

# Stage 2: Create the runtime image
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Copy the virtual environment with dependencies from the builder stage
COPY --from=builder /app/.venv ./.venv

# Add the virtual environment's bin directory to the PATH
ENV PATH="/app/.venv/bin:${PATH}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

# Copy the adapter application code and default voices file
COPY adapter.py voices.yml ./

# Expose the port the app runs on
EXPOSE 8004

# Run Uvicorn on container start
CMD ["uvicorn", "adapter:app", "--host", "0.0.0.0", "--port", "8004"]
