FROM python:3.10-slim as base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Stage 1: Heavy Builder (Weekly indexing & map generation)
# ==========================================
FROM base as heavy

# Install heavy dependencies
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Copy source code and data
COPY src/ ./src/
COPY study/ ./study/

# Create directory for artifacts
RUN mkdir -p /app/artifacts

# Command to run indexing script
CMD ["python", "src/main/generate_index_map.py"]

# ==========================================
# Stage 2: Light Runner (Daily Slack Alert)
# ==========================================
FROM base as light

# Install light dependencies
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt

# Copy source code (Light bot does not need data/index, only code)
COPY src/main/slack_bot_daily_review.py ./src/main/

# Command to run: Read mapped JSON and send to Slack
CMD ["python", "src/main/slack_bot_daily_review.py"]