# hex-prep container — GPU web UI + CLI. Needs an NVIDIA GPU and the
# NVIDIA Container Toolkit on the host.
# Build:  docker compose up --build      (or: docker build -t hex-prep .)
# Models cache in the /models volume, stems land in /output.
#
# Plain Ubuntu base, not nvidia/cuda: PyTorch's Linux wheels bundle their own
# CUDA libraries and the Container Toolkit injects the driver at run time, so
# the CUDA base image's ~4 GB of libraries went unused. Ubuntu 24.04 ships a
# real python3.12 (22.04's "python3.11" is 3.11.0rc1, which breaks onnx).

# ---- build stage --------------------------------------------------------
# diffq (an audio-separator dependency) has no prebuilt wheel for Python 3.12
# and compiles a C extension, so the compiler lives here and stays out of the
# final image.
FROM ubuntu:24.04 AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY hex_prep/ hex_prep/
# [cpu] only picks CPU onnxruntime — torch from PyPI is the CUDA build on
# Linux either way. The default chain is all torch Roformers; onnxruntime only
# runs the optional .onnx models (--karaoke-model mdx/mdx2), on CPU here,
# instead of onnxruntime-gpu dragging in a second CUDA runtime.
RUN pip install --no-cache-dir ".[cpu]" \
    && find /opt/venv -name __pycache__ -type d -prune -exec rm -rf {} +
# triton (~0.9 GB) only backs torch.compile, which audio-separator leaves off
# (use_torch_compile=False) and hex-prep never enables.
RUN pip uninstall -y triton

# ---- runtime stage ------------------------------------------------------
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HEXPREP_MODEL_DIR=/models \
    HEXPREP_OUTPUT_DIR=/output \
    HEXPREP_MUSIC_DIR=/music
VOLUME ["/models", "/output"]
EXPOSE 7870

CMD ["hex-prep-web", "--host", "0.0.0.0", "--port", "7870"]
