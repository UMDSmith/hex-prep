# hex-prep

**Drop a song in. Get clean vocals and clean music out.**

A multi-pass vocal extraction pipeline built on [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) and the UVR model ecosystem. One command (or one drag-and-drop) runs a chain of separation models that each fix the previous one's weakness — instead of you juggling models in a GUI.

```
song.mp3 ──▶ [Inst HQ 3] ──▶ vocals ──▶ [Kim FT polish] ──▶ [DeEcho-DeReverb] ──▶ VOCALS (clean, dry)
                   │                                                     
                   └────────▶ INSTRUMENTAL ─────────────────────────────▶ MUSIC (no vocals)
```

## Why a chain?

Single-model separation always leaks. This chain was assembled by A/B listening tests plus measured checks:

| Pass | Model | Job | Why this one |
|---|---|---|---|
| 1 | **UVR-MDX-NET Inst HQ 3** | vocals ⁄ instrumental split | Best instrumental quality in listening tests. Instrumental-focused, so rapped vocals survive by construction. |
| 1b | **MelBand Roformer Kim FT (unwa)** | strip instrumental bleed from the vocal stem | HQ3's vocal side is "mix minus instrumental" — anything it misses (sparse acoustic guitar, etc.) lands in the vocals. Re-separating with a vocal-focused model removes it: measured **-37.7 dB → -91.0 dB** of guitar bleed on the guitar-only intro of *Fast Car*, with the vocal level untouched. |
| 2 *(opt-in)* | **MelBand Roformer Karaoke (becruily)** | lead ⁄ backing vocal split | Best open lead/back model runnable in audio-separator (top open entry on the [MVSEP lead/back leaderboard](https://mvsep.com/quality_checker/leaderboard/lead_back_vocals)). Verified on *Bohemian Rhapsody*: harmony intro → backing, solo verse → lead, opera section → backing. Off by default — most tracks don't need it. |
| 3 | **UVR-DeEcho-DeReverb** (VR Arch) | dry the vocal | Best dereverb in listening tests. Critical if the vocal feeds RVC/voice conversion — RVC can't model reverb. |

All models download automatically on first use (~2 GB total for the default chain).

## Install

Requires Python 3.10+, `ffmpeg` in PATH, and ideally an NVIDIA GPU. The
separation stack (torch + onnxruntime) is heavy — install into its own
environment, not your system Python. Pick one:

### Option A — venv

```bash
git clone https://github.com/UMDSmith/hex-prep
cd hex-prep
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu]"     # or ".[cpu]" — much slower
```

### Option B — conda

```bash
git clone https://github.com/UMDSmith/hex-prep
cd hex-prep
conda create -n hex-prep python=3.11 -y
conda activate hex-prep
pip install -e ".[gpu]"     # or ".[cpu]"
```

(`requirements.txt` has the same dependencies if you prefer `pip install -r`.)

### Option C — Docker

No Python setup at all. Needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU access.

```bash
git clone https://github.com/UMDSmith/hex-prep
cd hex-prep
docker compose up --build     # web UI on http://localhost:7870
```

Model checkpoints persist in `./models`, stems land in `./output`. For CLI
runs inside the container:

```bash
docker compose run --rm hex-prep hex-prep --file /output/song.mp3
```

The AI models themselves (~2 GB for the default chain) download automatically
on first use in every install mode — no manual model setup.

### GPU support

Hardware detection is automatic at runtime — there is nothing to configure.
torch and onnxruntime probe for a CUDA device and quietly fall back to CPU
if there isn't one.

| Hardware | Install | What you get |
|---|---|---|
| NVIDIA (GTX/RTX, current driver) | `[gpu]` | Full acceleration, seconds per song |
| CPU only | `[cpu]` | Same results, minutes per song |
| Apple Silicon | `[cpu]` | torch-based passes accelerate via MPS automatically |
| AMD GPU | `[cpu]` | Runs on CPU — ROCm isn't supported by this stack |

For Docker without an NVIDIA GPU, delete the `gpus: all` line from
`docker-compose.yml` and it runs on CPU.

## Web UI

```bash
hex-prep-web                # http://127.0.0.1:7870
hex-prep-web --host 0.0.0.0 # reachable from other devices on your network
```

Drop an audio file, then pick a button:

- **Process** — download **clean vocals (.wav)** + **clean music (.mp3)**
- **Split backing vocals** — adds the karaoke pass and gives you three stems: **lead vocal**, **backing vocals**, and **instrumental** (all .wav). Use this for duets, choirs, and harmony-heavy tracks.

One GPU job runs at a time; extra uploads queue.

## CLI

```bash
hex-prep --file /path/to/song.flac          # process an exact file
hex-prep "tracy chapman fast car"           # search your music library
hex-prep --list "beatles"                   # preview matches, process nothing
hex-prep --karaoke "queen bohemian"         # also split lead vs backing vocals
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

- `<song>__LEAD_DRY.wav` — clean, dereverbed vocal (`__LEAD.wav` with `--no-dereverb`)
- `<song>__INSTRUMENTAL.wav` — the music, no vocals
- `<song>__MUSIC.mp3` — with `--music` (web UI always makes it): instrumental + backing in one file
- `<song>__BACKING.wav` / `__BACKING_DRY.wav` — only when `--karaoke` ran

### All flags

| Flag | What it does | When to use |
|---|---|---|
| `--file PATH` | Process an exact file | Song isn't in your library |
| `--list` | Show matches, don't process | Check the query first |
| `--karaoke` | Lead/backing split (becruily model) | Duets, harmony-heavy tracks, lead-only voice conversion |
| `--karaoke-model X` | Different split model: `becruily`, `viperx`, `gabox`, `gabox2`, `5hp`, `6hp`, `mdx`, `mdx2` | A/B testing when the default mangles a track |
| `--swap-lead` | Invert lead/backing assignment | Model picked the wrong vocal as lead |
| `--dereverb-backing` | Also dereverb the backing stem | You want both stems dry for layering |
| `--music` | Mix instrumental + backing into one MUSIC.mp3 | Karaoke/streaming setups |
| `--no-polish` | Skip the bleed-polish pass | Polish eats part of the vocal on a specific track |
| `--no-dereverb` | Skip the dereverb pass | Dereverb is destroying the vocal (can happen on rap) |
| `--dereverb-model X` | Different dereverb: `vr` (default), `anvuew`, `anvuew-less`, `big`, `super-big`, `echo`, `echo-fused`, `mono`, `bs` | Heavy reverb the default can't remove |
| `--dereverb-passes N` | Stack the dereverb N times (try 2-3) | Cathedral-level reverb |
| `--aggression N` | VR Arch aggression (default 5, try 10-15) | Push DeEcho-DeReverb (or 5hp/6hp) harder |
| `--keep-intermediate` | Keep every pass's stems | Debugging a weird result |

## Troubleshooting

- **Vocals still reverby** → `--dereverb-passes 2` or `--aggression 10`.
- **Dereverb destroyed the vocal** (common on rap) → `--no-dereverb`.
- **Target vocal missing from output** → `--keep-intermediate`, then listen to each pass's stems to find where it vanished.
- **Hangs on first run** → it's downloading models (100-900 MB each, one time).
- **Second web upload seems stuck** → one GPU job at a time; it's queued.

## Performance

On an RTX 5090: ~11-25 s per song for the default 3-pass chain; the karaoke pass adds ~20 s. CPU works but expect minutes per pass.

## Credits

- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (beveradb / Nomad Karaoke) does the actual inference.
- Models by the [UVR](https://github.com/Anjok07/ultimatevocalremovergui) community: Anjok07 & aufr33 (Inst HQ 3, DeEcho-DeReverb), Kimberley Jensen & unwa (Kim FT), becruily (karaoke), and the authors behind every `--*-model` shortcut.
- [MVSEP](https://mvsep.com) for the public quality leaderboards this chain was chosen against.

**Note:** the code here is MIT-licensed; the model checkpoints it downloads carry their own licenses (several are non-commercial). Check before commercial use.
