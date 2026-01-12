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

# Create directories for Volume Mounts (fixes "not found" error)
# We do NOT copy 'study' here. We mount it at runtime.
RUN mkdir -p /app/study /app/artifacts

# Copy source code
COPY src/ ./src/

# Command to run indexing script
# Ensure your script uses os.getenv('DATA_DIR', './study')
CMD ["python", "src/main/generate_index_map.py"]

# ==========================================
# Stage 2: Light Runner (Daily Slack Alert)
# ==========================================
FROM base as light

# Install light dependencies
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt

# Create artifacts directory (to read the json map)
RUN mkdir -p /app/artifacts

# Copy source code
COPY src/ ./src/

# Command to run: Read mapped JSON and send to Slack
CMD ["python", "src/main/slack_bot_daily_review.py"]