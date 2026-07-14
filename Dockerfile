# hex-prep container — GPU-ready web UI + CLI.
# Build:  docker compose up --build      (or: docker build -t hex-prep .)
# Models cache in the /models volume, stems land in /output.
#
# Base notes: Ubuntu 24.04 ships a real python3.12 (22.04's "python3.11" is
# 3.11.0rc1, which breaks onnx at runtime), and CUDA 12.8 is the floor for
# Blackwell (RTX 50-series) kernels in onnxruntime.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# build-essential + python3-dev: diffq (audio-separator dependency) has no
# prebuilt wheel for this python and compiles a C extension at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY hex_prep/ hex_prep/
RUN pip install --no-cache-dir ".[gpu]"

ENV HEXPREP_MODEL_DIR=/models \
    HEXPREP_OUTPUT_DIR=/output \
    HEXPREP_MUSIC_DIR=/music
VOLUME ["/models", "/output"]
EXPOSE 7870

CMD ["hex-prep-web", "--host", "0.0.0.0", "--port", "7870"]
