FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-tk \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    pkg-config \
    ffmpeg \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libegl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxi6 \
    libxcursor1 \
    libxkbcommon0 \
    libsm6 \
    libxmu6 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir jupyterlab ipykernel ipywidgets

COPY . /workspace

RUN if [ -f /workspace/asset/objaverse/plate_11.zip ]; then \
    unzip -o /workspace/asset/objaverse/plate_11.zip -d /workspace/asset/objaverse/; \
    fi

ENV MPLBACKEND=module://matplotlib_inline.backend_inline \
    MUJOCO_GL=egl

CMD ["bash"]