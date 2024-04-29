# syntax=docker/dockerfile:1.4

FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

# Install required system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    direnv \
    git \
    expect \
    && rm -rf /var/lib/apt/lists/*


# This cache directory will mounted as a volume inside the container to preserve
# it across rebuilds. Docker mounts "take" ownership and permissions from
# existing folders, so we create & chown ahead of time.
RUN mkdir /home/vscode/.cache && chown vscode:vscode /home/vscode/.cache
RUN pipx install pre-commit
RUN pipx install ruff==0.4.1
RUN pipx install poetry==1.6.0

# Ensure PYTHONPATH is set for streamlit
ENV PYTHONPATH=/workspace
# Prevents Streamlit from guessing address
ENV STREAMLIT_BROWSER_SERVER_ADDRESS=localhost
