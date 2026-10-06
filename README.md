# hex-prep

**Drop a song in. Get clean vocals and clean music out.**

> **Requires a capable NVIDIA GPU** — RTX 20-series or newer with 6 GB+ VRAM.
> See [GPU requirements](#gpu-requirements).

A multi-stage vocal extraction pipeline built on [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) and the UVR / MSST model ecosystem. One command (or one drag-and-drop) runs a chain of separation models that each do one job — instead of you juggling models in a GUI.

```
song.mp3 ─▶ [1 Leap Xe] ─▶ vocals ─▶ [2 BS Karaoke] ─▶ lead ─▶ [3 BS De-Reverb] ─▶ [4 De-Echo]* ─▶ LEAD_DRY
                │                          │
                │                          └─▶ BACKING ──┐
                └─▶ INSTRUMENTAL ────────────────────────┴─▶ MUSIC (instrumental + backing)
                                                                       * opt-in: --deecho
```

## Why a chain?

Single-model separation always leaks, and each job has its own specialist
model. The defaults (October 2026) are the best open models for each stage
per the [MVSEP leaderboards](https://mvsep.com/quality_checker/multisong_leaderboard?sort=vocals)
and the community's [separation guide](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c/mobilebasic):

| Stage | Model | Job | Why this one |
|---|---|---|---|
| 1 | **BS-Roformer Leap Xe** (unwa) | vocals ⁄ music split | Best public vocal model on MVSEP's multisong set: SDR 11.76, bleedless 37.4, fullness 18.3 — ahead of the Kim FT3 Preview it replaced on all three. |
| 2 | **BS-Roformer Karaoke** (becruily & frazer) | lead ⁄ backing vocal split | Community pick for harmony detection and telling lead from backing. Runs on the stage-1 vocals, so instruments can't confuse it. |
| 3 | **BS-Roformer De-Reverb, stereo** (anvuew) | dry the lead | SDR 22.51 vs 19.17 for anvuew's previous MelBand dereverb on the same validation set. Critical if the vocal feeds RVC/voice conversion — RVC can't model reverb. |
| 4 *(opt-in)* | **MelBand De-Reverb-Echo V2** (Sucial) | strip echo ⁄ delay | Best full-band open de-echo. Off by default: most songs have reverb but little delay, and every extra pass costs some fidelity. |

**Order matters:** the lead/backing split runs *before* dereverb. anvuew's
dereverb models also strip harmonies that sit off-center, and read backing
vocals as reverb — dereverbing first smears the lead.

All models download automatically on first use (~0.7 GB for the default
chain, ~1.1 GB with de-echo). Leap Xe and the BS dereverb aren't in audio-separator's model
registry, so hex-prep fetches them from Hugging Face itself (pinned
versions).

## Quick start

```bash
git clone https://github.com/UMDSmith/hex-prep
cd hex-prep
./start.sh        # Linux / macOS
start.bat         # Windows (double-click works too)
```

The first run creates `.venv/` inside the project folder and installs
everything into it — nothing touches your system Python (several GB, one
time). If the install gets interrupted, just run the script again and it picks up where it
left off. Every run after that starts the web UI and opens it in your
browser. Extra arguments pass through, e.g. `./start.sh --host 0.0.0.0`.

| Variable | Use |
|---|---|
| `HEXPREP_PYTHON` | Use an existing environment's python (one with hex-prep installed) instead of building `.venv/` |
| `HEXPREP_TORCH_INDEX` | PyTorch wheel index for the first-run install — see [GPU requirements](#gpu-requirements) |

## Install

### Prerequisites

- **Python 3.10–3.13** (3.12 recommended). The start scripts pick a
  supported version automatically if you have several.
- **ffmpeg** on your PATH (`sudo apt install ffmpeg`, `brew install ffmpeg`,
  `winget install ffmpeg`). It writes the music `.mp3` and decodes
  m4a/aac/wma input; wav, flac, mp3, ogg and opus are read directly.
- **Linux / macOS with Python 3.11+:** a C compiler and the Python headers.
  One dependency (diffq, via audio-separator) has no prebuilt wheel there, so
  pip compiles it: `sudo apt install build-essential python3-dev python3-venv`
  (Debian/Ubuntu), `sudo dnf install gcc python3-devel` (Fedora),
  `xcode-select --install` (macOS). `start.sh` checks for this up front.
  Windows needs nothing extra.
- **A capable NVIDIA GPU** — RTX 20-series or newer with 6 GB+ VRAM
  (hex-prep peaked at about 5 GB in testing). See
  [GPU requirements](#gpu-requirements).

The start scripts are the easy path. To manage the environment yourself:

### Option A — venv

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu]"
```

On **Windows with an NVIDIA card**, install CUDA torch first — PyPI's Windows
torch is CPU-only:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`

### Option B — conda

```bash
conda create -n hex-prep python=3.12 -y
conda activate hex-prep
pip install -e ".[gpu]"
```

(`requirements.txt` has the same dependencies if you prefer `pip install -r`.)

### Option C — Docker

No Python setup at all — the most isolated option (Linux + NVIDIA). Needs
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU access.
The image is about 9 GB, nearly all of it PyTorch and its CUDA libraries.

```bash
docker compose up --build     # web UI on http://localhost:7870
```

Model checkpoints persist in `./models`, stems land in `./output`. For CLI
runs inside the container:

```bash
docker compose run --rm hex-prep hex-prep --file /output/song.mp3
```

The AI models themselves download automatically on first use in every
install mode — no manual model setup.

### GPU requirements

hex-prep requires an NVIDIA GPU. The GPU is picked up automatically — there
is nothing to configure. Without one, torch falls back to the CPU, which is
far too slow to be practical; Apple Silicon and AMD GPUs aren't supported.

| GPU | Notes |
|---|---|
| RTX 20-series or newer, 6 GB+ VRAM | Supported. On Linux, PyPI's torch is built for CUDA 13 and needs NVIDIA driver **580+**; Windows `start.bat` installs the CUDA 12.8 build (driver 570+). |
| GTX 10-series, or a Linux driver older than 580 | Untested. Set `HEXPREP_TORCH_INDEX=https://download.pytorch.org/whl/cu126` before the first `start` run (or `pip install torch --index-url` that URL before installing hex-prep). |

## Web UI

```bash
hex-prep-web                # http://127.0.0.1:7870
hex-prep-web --host 0.0.0.0 # reachable from other devices on your network
```

Drop an audio file and hit **Process** to download **clean vocals (.wav)**
(lead only, dry) + **clean music (.mp3)** (instrumental with the backing
vocals mixed back in). Tick **Remove echo / delay** to add stage 4.

One GPU job runs at a time; extra uploads queue.

## CLI

```bash
hex-prep --file /path/to/song.flac          # process an exact file
hex-prep "tracy chapman fast car"           # search your music library
hex-prep --list "beatles"                   # preview matches, process nothing
hex-prep --deecho "queen bohemian"          # add stage 4: strip echo/delay
hex-prep --music --file song.mp3            # also produce a single MUSIC.mp3
```

Library search: every word in the query must appear somewhere in the file path (case-insensitive). One match processes immediately; several gives you a numbered picker.

### Configuration

| Env var | Meaning | Default |
|---|---|---|
| `HEXPREP_MUSIC_DIR` | music library for query search | `~/Music` |
| `HEXPREP_OUTPUT_DIR` | where stems are written | `~/hex-prep-output` |
| `HEXPREP_MODEL_DIR` | model checkpoint cache | `~/audio-separator-models` |

### Outputs

`$HEXPREP_OUTPUT_DIR/<song name>/`

- `<song>__LEAD_DRY.wav` — clean, dry lead vocal (`__LEAD.wav` with `--no-dereverb` and no `--deecho`)
- `<song>__INSTRUMENTAL.wav` — the music, no vocals
- `<song>__BACKING.wav` — backing vocals (`__BACKING_DRY.wav` with `--dereverb-backing`; absent with `--no-karaoke`)
- `<song>__MUSIC.mp3` — with `--music` (web UI always makes it): instrumental + backing in one file

### All flags

| Flag | What it does | When to use |
|---|---|---|
| `--file PATH` | Process an exact file | Song isn't in your library |
| `--list` | Show matches, don't process | Check the query first |
| `--vocal-model X` | Different stage-1 model: `leap-xe` (default), `bigbeta6x`, `ft3` | BACKING comes out thin (Leap Xe sometimes leaves backing vocals in the instrumental — try `bigbeta6x`) |
| `--no-karaoke` | Skip the lead/backing split | Solo vocal, or you want every vocal in LEAD |
| `--karaoke-model X` | Different split model: `frazer` (default), `anvuew`, `becruily`, `gabox`, `gabox2`, `viperx`, `5hp`, `6hp`, `mdx`, `mdx2` | A/B testing when the default mangles a track (`anvuew` gives a brighter lead and handles radio-effect vocals) |
| `--swap-lead` | Invert lead/backing assignment | Model picked the wrong vocal as lead |
| `--dereverb-backing` | Also dry the backing stem (dereverb, plus de-echo if on) | You want both stems dry for layering |
| `--music` | Mix instrumental + backing into one MUSIC.mp3 | Karaoke/streaming setups |
| `--no-dereverb` | Skip the dereverb pass | Dereverb is destroying the vocal (can happen on rap) |
| `--dereverb-model X` | Different dereverb: `anvuew` (default, BS stereo), `anvuew-mel`, `anvuew-less`, `mono`, `big`, `super-big`, `echo`, `echo-fused`, `vr`, `bs` | Heavy reverb the default can't remove |
| `--dereverb-passes N` | Stack the dereverb N times (try 2-3) | Cathedral-level reverb |
| `--deecho` | Add stage 4: de-echo the lead after dereverb | Audible delay/echo repeats on the vocal |
| `--deecho-model X` | Different de-echo (implies `--deecho`): `sucial` (default), `sucial-fused`, `vr-normal`, `vr-aggressive` | Heavy delay the default leaves behind (`vr-*` tops out ~17.5 kHz) |
| `--aggression N` | VR Arch aggression (default 5, try 10-15) | Push the `vr*` dereverb/de-echo (or 5hp/6hp) harder |
| `--overlap N` | Overlapping windows per Roformer pass (default 16) | `4` is ~4x faster for ~0.01 dB less SDR (see Performance) |
| `--keep-intermediate` | Keep every pass's stems | Debugging a weird result |

## Troubleshooting

- **Vocals still reverby** → `--dereverb-passes 2`.
- **Echo/delay repeats left on the vocal** → `--deecho`, then `--deecho-model vr-aggressive` if it persists.
- **Backing vocals missing from BACKING** → `--vocal-model bigbeta6x`.
- **Dereverb destroyed the vocal** (common on rap) → `--no-dereverb`.
- **Target vocal missing from output** → `--keep-intermediate`, then listen to each pass's stems to find where it vanished.
- **Hangs on first run** → it's downloading models (200-460 MB each, one time).
- **Too slow** → `--overlap 4` (see Performance).
- **Second web upload seems stuck** → one GPU job at a time; it's queued.

## Performance

Measured on one RTX 5090 with a 3.5-minute song:

| | `--overlap 16` (default) | `--overlap 4` |
|---|---|---|
| Stage 1 — Leap Xe | 163 s | 43 s |
| Stage 2 — BS karaoke | 87 s | 22 s |
| Stage 3 — BS dereverb | 74 s | 18 s |
| **Default chain total** | **~5.5 min** | **83 s** |
| Stage 4 — de-echo (`--deecho`) | +62 s | — |

Pass time scales linearly with `--overlap`. On the same song, Leap Xe's
vocal SDR was 15.65 / 15.68 / 15.69 dB at overlap 2 / 4 / 16 — so 4 gets
essentially all the quality at a quarter of the time. `--dereverb-backing`
repeats stages 3-4 on the backing stem.

## Credits

- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (beveradb / Nomad Karaoke) does the actual inference.
- Models by the UVR / MSST community: unwa ([Leap Xe](https://huggingface.co/pcunwa/BS-Roformer-Leap), Big Beta 6X, Kim FT3), becruily & frazer (BS karaoke), anvuew ([BS dereverb](https://huggingface.co/anvuew/dereverb_bs_roformer), BS karaoke), Sucial (de-reverb-echo), FoxJoy, aufr33, viperx, Gabox, and the authors behind every `--*-model` shortcut.
- [MVSEP](https://mvsep.com) for the public quality leaderboards and deton24's community guide for the per-stage rankings this chain was chosen against.

**Note:** the code here is MIT-licensed. hex-prep doesn't bundle any model weights — it downloads them from their authors' repos on first use, and they carry their own licenses (e.g. anvuew's BS dereverb is GPL-3.0; unwa's Leap Xe states none; several UVR models are non-commercial). Check before commercial use.
