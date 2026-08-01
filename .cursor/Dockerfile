# Cloud Agent base image for NumPy development (Ubuntu).
# Do not COPY the project — Cursor checks out the repo into the workspace.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CCACHE_DIR=/home/ubuntu/.ccache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ccache \
    curl \
    ca-certificates \
    gfortran \
    git \
    libopenblas-dev \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Prefer ccache for incremental C/Cython rebuilds across install checkpoints.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && ln -sf /usr/bin/ccache /usr/local/bin/gcc \
    && ln -sf /usr/bin/ccache /usr/local/bin/g++ \
    && ln -sf /usr/bin/ccache /usr/local/bin/cc \
    && ln -sf /usr/bin/ccache /usr/local/bin/c++

# Non-root user commonly used by Cloud Agents.
RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash ubuntu \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu
WORKDIR /home/ubuntu
