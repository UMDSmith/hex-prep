#!/usr/bin/env bash
# Launch the hex-prep web UI and open it in your browser.
#
# First run creates .venv/ and installs hex-prep into it (pulls PyTorch,
# several GB). Optional environment variables:
#   HEXPREP_PYTHON       use this interpreter (an env that already has
#                        hex-prep) instead of building .venv/
#   HEXPREP_TORCH_INDEX  PyTorch wheel index for the first-run install, e.g.
#                        https://download.pytorch.org/whl/cu126 for GTX
#                        10-series cards or NVIDIA drivers older than 580
# Extra arguments go to hex-prep-web, e.g.  ./start.sh --host 0.0.0.0
set -euo pipefail
cd "$(dirname "$0")"

# Written once the install finishes, so a first run that was interrupted
# (or failed) resumes the install instead of launching a half-built .venv.
STAMP=.venv/.hexprep-installed

find_python() {
    # 3.10-3.13: the versions the dependency stack ships wheels for.
    for py in python3.12 python3.11 python3.13 python3.10 python3; do
        command -v "$py" >/dev/null || continue
        if "$py" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null; then
            echo "$py"
            return
        fi
    done
}

install() {
    if [ ! -x .venv/bin/python ]; then
        local py
        py=$(find_python)
        if [ -z "$py" ]; then
            echo "Need Python 3.10-3.13 (3.12 recommended)." >&2
            exit 1
        fi
        if ! "$py" -m venv .venv; then
            rm -rf .venv
            echo "Couldn't create a virtualenv. On Debian/Ubuntu: sudo apt install $(basename "$py")-venv" >&2
            exit 1
        fi
    fi
    local vpy=.venv/bin/python

    # diffq (an audio-separator dependency) ships Linux/macOS wheels only up
    # to Python 3.10; newer Pythons compile it, which needs a C compiler and
    # the Python headers. Check now rather than fail deep inside pip.
    if ! "$vpy" -c 'import sys; sys.exit(sys.version_info[:2] > (3, 10))'; then
        if ! command -v cc >/dev/null || ! "$vpy" -c 'import os, sysconfig
raise SystemExit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))'; then
            echo "A dependency needs a C compiler and Python headers to build. Install them, then re-run:" >&2
            echo "  Debian/Ubuntu:  sudo apt install build-essential python3-dev" >&2
            echo "  Fedora:         sudo dnf install gcc python3-devel" >&2
            echo "  macOS:          xcode-select --install" >&2
            exit 1
        fi
    fi

    command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not in PATH — the music mix (.mp3) will be skipped." >&2
    local extra=cpu
    if command -v nvidia-smi >/dev/null; then
        extra=gpu
    else
        echo "WARNING: no NVIDIA GPU found. hex-prep needs one — it will install, but runs far too slowly on CPU." >&2
    fi
    echo "First run: installing hex-prep[$extra] into .venv (several GB)..."
    "$vpy" -m pip install --upgrade pip
    if [ -n "${HEXPREP_TORCH_INDEX:-}" ]; then
        "$vpy" -m pip install torch --index-url "$HEXPREP_TORCH_INDEX"
    fi
    "$vpy" -m pip install -e ".[$extra]"
    touch "$STAMP"
}

PY="${HEXPREP_PYTHON:-}"
if [ -z "$PY" ]; then
    [ -f "$STAMP" ] || install
    PY=.venv/bin/python
fi

exec "$PY" -m hex_prep.web --open "$@"
