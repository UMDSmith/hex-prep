"""
hex-prep CLI — multi-stage vocal extraction for voice conversion / karaoke stems.

Stage 1:  unwa BS-Roformer Leap Xe               -> vocals / music
Stage 2:  becruily & frazer BS-Roformer karaoke  -> lead / backing vocals
Stage 3:  anvuew BS-Roformer dereverb (stereo)   -> dry the lead
Stage 4:  Sucial MelBand de-reverb-echo V2       -> strip echo/delay (opt-in)

Stages 1-3 run by default; stage 4 with --deecho. Order is load-bearing:
backing vocals must come out BEFORE dereverb — anvuew's dereverb models also
strip harmonies that sit off-center, and otherwise read the backing as reverb
content and smear the lead. De-echo runs last, on an already-dry vocal.
Skip stages with --no-karaoke / --no-dereverb.

Processes a library search hit from your music folder, or an explicit file
path. Prints final stem paths. Also importable: process() returns a dict of
the final output paths (used by the web UI).

Configuration (environment variables):
    HEXPREP_MUSIC_DIR   music library for query search   (default: ~/Music)
    HEXPREP_OUTPUT_DIR  where stems are written          (default: ~/hex-prep-output)
    HEXPREP_MODEL_DIR   model checkpoint cache           (default: ~/audio-separator-models)

Usage:
    hex-prep "2pac changes"
    hex-prep --list "beatles"
    hex-prep --file /path/to/song.mp3 --deecho
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
import soundfile as sf
from audio_separator.separator import Separator
from tqdm import tqdm

# ----- Configuration -------------------------------------------------------

MUSIC_DIR = Path(os.getenv("HEXPREP_MUSIC_DIR", str(Path.home() / "Music")))
OUTPUT_DIR = Path(os.getenv("HEXPREP_OUTPUT_DIR", str(Path.home() / "hex-prep-output")))
MODEL_DIR = Path(os.getenv("HEXPREP_MODEL_DIR", str(Path.home() / "audio-separator-models")))
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".wma"}

# Default chain (2026-10), picked from the MVSEP leaderboards and deton24's
# community separation guide. Vocal metrics are MVSEP multisong
# SDR / bleedless / fullness.
#   isolate  — unwa BS-Roformer Leap Xe, vocal target: 11.76 / 37.4 / 18.3,
#              the best public vocal model, and ahead of the Kim FT3 Preview
#              it replaces (11.05 / 36.1 / 16.8) on all three. It sometimes
#              leaves backing vocals in the instrumental — harmless for the
#              music mix; try `--vocal-model bigbeta6x` if BACKING comes out
#              thin. Not in audio-separator's registry: see EXTRA_MODELS.
#   karaoke  — becruily & frazer BS-Roformer karaoke: the community's pick
#              for harmony detection and telling lead from backing (MVSEP
#              lead SDR 10.10). anvuew's BS karaoke scores 10.23 with a
#              brighter lead — `--karaoke-model anvuew`.
#   dereverb — anvuew BS-Roformer dereverb, stereo: SDR 22.51 vs 19.17 for
#              his MelBand v2 (the previous default) on the same validation
#              set. Ignores --aggression (VR-only); stack --dereverb-passes
#              for heavy verb instead. Not in the registry: see EXTRA_MODELS.
#   deecho   — Sucial MelBand de-reverb-echo V2. No open model rivals paid
#              RX 11 Dialogue Isolate for delay; this is the best full-band
#              one. Opt-in: most songs carry reverb but little delay, and
#              every extra pass costs some fidelity.
MODELS = {
    "isolate":  "bs_roformer_leap_xe_voc_unwa.ckpt",
    "karaoke":  "bs_roformer_karaoke_frazer_becruily.ckpt",
    "dereverb": "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
    "deecho":   "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
}

# Checkpoints audio-separator's registry doesn't carry. Both use standard
# BS-Roformer configs its loader handles; HexSeparator lists them and fetches
# the files from Hugging Face (pinned commits) on first use. Local config
# names keep "roformer" in them — audio-separator's fallback roformer
# detection keys on the yaml filename.
_HF = "https://huggingface.co"
_LEAP = f"{_HF}/pcunwa/BS-Roformer-Leap/resolve/4e47d6662ae82eaa8b4ac4329fe66099a843b48e"
_ANVUEW_DV = f"{_HF}/anvuew/dereverb_bs_roformer/resolve/bd5c6b55a429b4b74ce85fe5dcb690cfe36d91ec"
EXTRA_MODELS = {
    "bs_roformer_leap_xe_voc_unwa.ckpt": {
        "name": "BS Roformer | Leap Xe Vocals by unwa",
        "url": f"{_LEAP}/Xe/bs_leap_xe_voc.ckpt",
        "config": "config_bs_roformer_leap_xe_voc_unwa.yaml",
        "config_url": f"{_LEAP}/Xe/leap_xe_config_voc.yaml",
    },
    "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt": {
        "name": "BS Roformer | De-Reverb stereo by anvuew",
        "url": f"{_ANVUEW_DV}/dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
        "config": "config_dereverb_bs_roformer_anvuew_sdr_22.5050.yaml",
        "config_url": f"{_ANVUEW_DV}/config.yaml",
    },
}

VOCAL_SHORTCUTS = {
    "leap-xe":   "bs_roformer_leap_xe_voc_unwa.ckpt",
    "bigbeta6x": "melband_roformer_big_beta6x.ckpt",
    "ft3":       "mel_band_roformer_kim_ft3_unwa.ckpt",
}

KARAOKE_SHORTCUTS = {
    "frazer":   "bs_roformer_karaoke_frazer_becruily.ckpt",
    "anvuew":   "bs_roformer_karaoke_anvuew.ckpt",
    "becruily": "mel_band_roformer_karaoke_becruily.ckpt",
    "gabox":    "mel_band_roformer_karaoke_gabox.ckpt",
    "gabox2":   "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "viperx":   "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "5hp":      "5_HP-Karaoke-UVR.pth",
    "6hp":      "6_HP-Karaoke-UVR.pth",
    "mdx":      "UVR_MDXNET_KARA.onnx",
    "mdx2":     "UVR_MDXNET_KARA_2.onnx",
}

DEREVERB_SHORTCUTS = {
    "anvuew":      "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
    "anvuew-mel":  "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "anvuew-less": "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    "mono":        "dereverb_mel_band_roformer_mono_anvuew.ckpt",
    "big":         "dereverb_big_mbr_ep_362.ckpt",
    "super-big":   "dereverb_super_big_mbr_ep_346.ckpt",
    "echo":        "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    "echo-fused":  "dereverb_echo_mbr_fused.ckpt",
    "vr":          "UVR-DeEcho-DeReverb.pth",
    "bs":          "deverb_bs_roformer_8_384dim_10depth.ckpt",
}

DEECHO_SHORTCUTS = {
    "sucial":        "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    "sucial-fused":  "dereverb_echo_mbr_fused.ckpt",
    "vr-normal":     "UVR-De-Echo-Normal.pth",
    "vr-aggressive": "UVR-De-Echo-Aggressive.pth",
}

# ----- Library search ------------------------------------------------------

def find_songs(query: str, music_dir: Path = MUSIC_DIR) -> list[Path]:
    """Return audio files whose path contains all whitespace-separated tokens
    in `query` (case-insensitive). Cheap and effective for a flat-ish library."""
    tokens = [t.lower() for t in query.split() if t]
    if not tokens:
        return []
    matches = []
    for path in music_dir.rglob("*"):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        haystack = str(path).lower()
        if all(tok in haystack for tok in tokens):
            matches.append(path)
    matches.sort()
    return matches


def pick_song(matches: list[Path]) -> Path | None:
    """Interactive picker. Returns chosen path or None if user bails."""
    if not matches:
        print("No matches found.")
        return None
    if len(matches) == 1:
        return matches[0]

    print(f"\n{len(matches)} matches:\n")
    # Cap display so we don't spam terminal on broad queries
    display_limit = 50
    shown = matches[:display_limit]
    for i, p in enumerate(shown, 1):
        try:
            rel = p.relative_to(MUSIC_DIR)
        except ValueError:
            rel = p
        print(f"  [{i:>3}] {rel}")
    if len(matches) > display_limit:
        print(f"  ... and {len(matches) - display_limit} more (refine your query)")

    while True:
        choice = input("\nPick a number (or 'q' to quit): ").strip()
        if choice.lower() in {"q", "quit", "exit", ""}:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(shown):
            return shown[int(choice) - 1]
        print("Invalid choice.")


# ----- Separator helpers ---------------------------------------------------

def fetch(url: str, dest: Path) -> None:
    """Download `url` to `dest` unless it's already there. Writes to a .part
    file first so an interrupted download can't pass for a finished one
    (audio-separator's own downloader writes in place)."""
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  Downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


class HexSeparator(Separator):
    """Separator that also knows EXTRA_MODELS. audio-separator only loads
    models from its registry listing, so add ours to the listing and put
    their files on disk before the base class looks for them (it skips the
    download when a file already exists)."""

    def list_supported_model_files(self):
        models = super().list_supported_model_files()
        for ckpt, info in EXTRA_MODELS.items():
            models["MDXC"][f"Roformer Model: {info['name']}"] = {
                "filename": ckpt, "scores": {}, "stems": [], "target_stem": None,
                "download_files": [ckpt, info["config"]],
            }
        return models

    def download_model_files(self, model_filename):
        info = EXTRA_MODELS.get(model_filename)
        if info:
            model_dir = Path(self.model_file_dir)
            fetch(info["config_url"], model_dir / info["config"])
            fetch(info["url"], model_dir / model_filename)
        return super().download_model_files(model_filename)


def make_separator(work_dir: Path, vr_aggression: int = 5,
                   overlap: int = 16) -> Separator:
    """One Separator instance reused across all passes — model swap is
    cheap, separator init is not. vr_aggression controls how hard VR Arch
    models push to remove the target stem (5 default, 10-15 for cleaner
    karaoke at risk of artifacts)."""
    return HexSeparator(
        output_dir=str(work_dir),
        model_file_dir=str(MODEL_DIR),
        output_format="WAV",
        sample_rate=44100,
        vr_params={"aggression": vr_aggression},
        # Every default stage is a Roformer. overlap is the number of
        # overlapping prediction windows (hop = chunk // overlap), so pass
        # time scales linearly with it. Measured on a 3.5 min test mix
        # (2026-10), Leap Xe stage-1 vocal SDR: 15.65 @2 (21s), 15.68 @4
        # (39s), 15.69 @16 (163s) — past 4 it buys time, not quality.
        # Must be the full dict: audio-separator stores it wholesale, no merge.
        mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                     "batch_size": 1, "overlap": overlap, "pitch_shift": 0},
    )


def run_pass(sep: Separator, label: str, model: str, input_path: Path) -> list[Path]:
    """Load `model`, separate `input_path`, return the resulting output paths."""
    print(f"\n--- {label} ---")
    print(f"  Model: {model}")
    print(f"  Input: {input_path.name}")
    t0 = time.time()
    sep.load_model(model_filename=model)
    output_files = sep.separate(str(input_path))
    elapsed = time.time() - t0
    # audio-separator returns relative filenames; resolve against output_dir
    out_dir = Path(sep.output_dir)
    resolved = [out_dir / f for f in output_files]
    print(f"  Done in {elapsed:.1f}s  ->  {[p.name for p in resolved]}")
    return resolved


def stash(path: Path, name: str) -> Path:
    """Rename a pass output to a short fixed name in the same folder.
    audio-separator appends '_(stem)_model' to every output, so chaining
    outputs straight into the next pass grows filenames past the 255-byte
    limit — and short inputs keep each pass's stem tag the only one."""
    dest = path.with_name(name)
    path.replace(dest)
    return dest


PAREN_RE = re.compile(r"\(([^)]+)\)")

# Stem tag names come from each model's config, and they differ per model:
# BS Roformer karaoke tags "(Vocals)"/"(Instrumental)", Leap Xe and the Kim
# MelBands tag "(vocals)"/"(other)". Accept known synonyms so a model swap
# doesn't break stem picking.
STEM_TAG_SYNONYMS = {
    "vocals": {"vocals"},
    "instrumental": {"instrumental", "other", "no vocals", "no_vocals"},
}

# Tags dereverb / de-echo models give their dry output: anvuew "noreverb",
# Sucial "dry" (its wet stem is "No dry"), FoxJoy VR "No Reverb"/"No Echo".
DRY_TAGS = {"noreverb", "no reverb", "no_reverb", "dry", "no echo", "noecho"}


def pick_stem(paths: list[Path], wanted: str) -> Path:
    """audio-separator names outputs with the stem type in parentheses, e.g.
    'foo_(Vocals)_BS-Roformer.wav'. Song names can carry their own
    parentheses ('Song (Live)'), so match the LAST parenthesized tag — the
    one this pass produced — before falling back to looser matches."""
    wanted_lower = wanted.lower()
    accepted = STEM_TAG_SYNONYMS.get(wanted_lower, set()) | {wanted_lower}

    for p in paths:
        tags = PAREN_RE.findall(p.name.lower())
        if tags and tags[-1] in accepted:
            return p

    # Fallback 1: any parenthesized tag matches
    for p in paths:
        tags = PAREN_RE.findall(p.name.lower())
        if accepted & set(tags):
            return p

    # Fallback 2: substring match without parens (last resort)
    for p in paths:
        if wanted_lower in p.name.lower():
            return p

    raise RuntimeError(
        f"Couldn't find a '{wanted}' stem in outputs: {[p.name for p in paths]}"
    )


def pick_dry(paths: list[Path]) -> Path:
    """Pick the dry stem from a dereverb / de-echo pass. Inputs to these
    passes are always stash()ed short names, so the last tag is this pass's."""
    for p in paths:
        tags = PAREN_RE.findall(p.name.lower())
        if tags and tags[-1] in DRY_TAGS:
            return p
    print(f"  WARN: couldn't auto-detect dry stem, defaulting to {paths[0].name}")
    return paths[0]


# ----- Main pipeline -------------------------------------------------------

def decodable_input(song: Path, work_dir: Path) -> Path:
    """Return a path audio-separator can read. librosa 1.0 loads audio only
    through soundfile (WAV, FLAC, OGG/Opus, MP3) — its audioread fallback is
    gone — so decode anything else (m4a, aac, wma, ...) to WAV with ffmpeg."""
    try:
        sf.info(str(song))
        return song
    except RuntimeError:  # soundfile.LibsndfileError: format not supported
        pass
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"Reading {song.suffix} files needs ffmpeg in PATH")
    wav = work_dir / "_input.wav"
    print(f"\n--- Decoding {song.suffix} input with ffmpeg ---")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(song),
         "-vn", "-c:a", "pcm_f32le", str(wav)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't decode {song.name}:\n{result.stderr}")
    return wav


def mix_to_music(instrumental: Path, backing: Path | None,
                 output: Path) -> Path | None:
    """Mix the instrumental + backing vocals into a single music.mp3 file
    for streaming/karaoke workflows. If backing is None or empty, just
    transcodes the instrumental to MP3. Uses ffmpeg via subprocess."""
    print(f"\n--- Mixing music track ---")
    print(f"  Instrumental: {instrumental.name}")
    if backing and backing.exists():
        print(f"  Backing:      {backing.name}")
    else:
        print(f"  Backing:      (none)")
    print(f"  Output:       {output.name}")

    if not shutil.which("ffmpeg"):
        print(f"  ERROR: ffmpeg not found in PATH; skipping music mix")
        return None

    t0 = time.time()

    if backing and backing.exists():
        # Mix two streams together at equal weights, output as MP3 at 192k
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(instrumental),
            "-i", str(backing),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0",
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(output),
        ]
    else:
        # Just transcode the instrumental to MP3
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(instrumental),
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(output),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: ffmpeg failed:\n{result.stderr}")
        return None

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return output


def process(song: Path, keep_intermediate: bool = False,
            swap_lead: bool = False,
            dereverb_backing: bool = False,
            no_karaoke: bool = False,
            karaoke_model_override: str | None = None,
            vr_aggression: int = 5,
            no_dereverb: bool = False,
            mix_music: bool = False,
            dereverb_model_override: str | None = None,
            dereverb_passes: int = 1,
            vocal_model_override: str | None = None,
            deecho: bool = False,
            deecho_model_override: str | None = None,
            overlap: int = 16) -> dict[str, Path]:
    """Run the extraction chain on `song`. Returns final output paths keyed
    by stem name ("lead", "instrumental", "backing", "backing_dry", "music" —
    only keys that were actually produced)."""
    models = dict(MODELS)
    chain_label = "LeapXe > BS-Karaoke > BS-Dereverb"

    if vocal_model_override:
        models["isolate"] = vocal_model_override
        chain_label += f" / VOCALS={vocal_model_override[:30]}"

    # Asking for a specific model means you want that stage, so each
    # override also cancels the matching --no-* / enables --deecho.
    if karaoke_model_override:
        models["karaoke"] = karaoke_model_override
        no_karaoke = False
        chain_label += f" / KARAOKE={karaoke_model_override[:30]}"
    if dereverb_model_override:
        models["dereverb"] = dereverb_model_override
        no_dereverb = False
        chain_label += f" / DEREVERB={dereverb_model_override[:30]}"
    if deecho_model_override:
        models["deecho"] = deecho_model_override
        deecho = True
        chain_label += f" / DEECHO={deecho_model_override[:30]}"
    elif deecho:
        chain_label += " > De-echo"

    if no_karaoke:
        chain_label += " / NO-KARAOKE"
    if no_dereverb:
        chain_label += " / NO-DEREVERB"
    if dereverb_passes > 1:
        chain_label += f" / DV-PASSES={dereverb_passes}"
    if swap_lead:
        chain_label += " / SWAP-LEAD"
    if vr_aggression != 5:
        chain_label += f" / AGGR={vr_aggression}"
    if overlap != 16:
        chain_label += f" / OVERLAP={overlap}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = OUTPUT_DIR / song.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Hex Prep ({chain_label})")
    print(f"  Source: {song}")
    print(f"  Output: {work_dir}")
    print(f"{'='*60}")

    sep = make_separator(work_dir, vr_aggression=vr_aggression, overlap=overlap)
    pipeline_start = time.time()

    # Stage 1: vocals / music
    source = decodable_input(song, work_dir)
    p1 = run_pass(sep, "Stage 1: vocals / music", models["isolate"], source)
    vocals = pick_stem(p1, "Vocals")
    instrumental = pick_stem(p1, "Instrumental")
    vocals = stash(vocals, "_s1_vocals.wav")
    instrumental = stash(instrumental, "_s1_instrumental.wav")

    # Stage 2: split lead from backing. Must run BEFORE dereverb (see the
    # module docstring).
    if no_karaoke:
        print("\n--- Skipping stage 2, lead/backing split (--no-karaoke) ---")
        lead, backing = vocals, None
    else:
        p2 = run_pass(sep, "Stage 2: lead / backing", models["karaoke"], vocals)
        # The karaoke models we ship tag the lead "(Vocals)" and the backing
        # "(Instrumental)" (the BS karaoke configs name them exactly that).
        # --swap-lead inverts this for the rare model/track where it's wrong.
        lead_tag, backing_tag = (("Instrumental", "Vocals") if swap_lead
                                 else ("Vocals", "Instrumental"))
        lead = pick_stem(p2, lead_tag)
        backing = pick_stem(p2, backing_tag)
        lead = stash(lead, "_s2_lead.wav")
        backing = stash(backing, "_s2_backing.wav")

    # Stages 3-4: dereverb, then de-echo.
    dried = not no_dereverb or deecho
    if no_dereverb:
        print("\n--- Skipping stage 3, dereverb (--no-dereverb) ---")

    def dry(stem: Path, who: str) -> Path:
        """Run stages 3-4 on one vocal stem. Stacked dereverb passes are the
        way to handle heavy verb — the Roformer dereverbs expose no
        aggressiveness knob."""
        if not no_dereverb:
            for n in range(1, dereverb_passes + 1):
                label = f"Stage 3: dereverb {who}"
                if dereverb_passes > 1:
                    label += f" (pass {n}/{dereverb_passes})"
                out = run_pass(sep, label, models["dereverb"], stem)
                stem = stash(pick_dry(out), f"_s3_{who}_dry{n}.wav")
        if deecho:
            out = run_pass(sep, f"Stage 4: de-echo {who}", models["deecho"], stem)
            stem = stash(pick_dry(out), f"_s4_{who}_dry.wav")
        return stem

    lead = dry(lead, "lead")
    if not deecho:
        print("\n--- Skipping stage 4, de-echo (enable with --deecho) ---")
    # Optional: dry the backing too (useful when the karaoke model put the
    # wrong vocal in backing, or you want both stems clean for layering)
    backing_dried = dereverb_backing and backing is not None and dried
    if backing_dried:
        backing = dry(backing, "backing")

    # Rename final outputs to predictable names
    final_lead = work_dir / f"{song.stem}__{'LEAD_DRY' if dried else 'LEAD'}.wav"
    final_inst = work_dir / f"{song.stem}__INSTRUMENTAL.wav"
    lead.replace(final_lead)
    instrumental.replace(final_inst)
    final_backing = None
    if backing is not None:
        final_backing = work_dir / f"{song.stem}__{'BACKING_DRY' if backing_dried else 'BACKING'}.wav"
        backing.replace(final_backing)

    # Optional: mix instrumental + backing into a single music file (for
    # workflows where the vocal stem drives something like lip-sync and one
    # music file plays everything else).
    final_music = None
    if mix_music:
        final_music = mix_to_music(
            instrumental=final_inst,
            backing=final_backing,
            output=work_dir / f"{song.stem}__MUSIC.mp3",
        )

    # Cleanup intermediates unless requested otherwise
    if not keep_intermediate:
        keep = {final_lead, final_inst, final_backing}
        for p in work_dir.glob("*.wav"):
            if p not in keep:
                p.unlink()

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Done in {total:.1f}s")
    print(f"  Lead vocal:        {final_lead}")
    print(f"  Instrumental:      {final_inst}")
    if final_backing is not None:
        print(f"  Backing vocals:    {final_backing}")
    if final_music is not None:
        print(f"  Music mix (.mp3):  {final_music}")
    print(f"{'='*60}\n")

    results: dict[str, Path] = {"lead": final_lead, "instrumental": final_inst}
    if final_backing is not None:
        results["backing_dry" if backing_dried else "backing"] = final_backing
    if final_music is not None:
        results["music"] = final_music
    return results


# ----- CLI -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*",
                    help="Search terms to find a song in HEXPREP_MUSIC_DIR")
    ap.add_argument("--file", type=str, default=None, metavar="PATH",
                    help="Process this exact audio file instead of searching "
                         "the music library")
    ap.add_argument("--list", action="store_true",
                    help="List matches without processing")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep intermediate stems instead of cleaning up")
    ap.add_argument("--vocal-model", type=str, default=None, metavar="MODEL",
                    help="Stage 1 vocal/music model. Shortcuts: 'leap-xe' "
                         "(default), 'bigbeta6x', 'ft3'. Or a full model "
                         "filename. (Models download automatically.)")
    ap.add_argument("--swap-lead", action="store_true",
                    help="Invert lead/backing assignment (use when the karaoke "
                         "model picks the wrong vocal as lead)")
    ap.add_argument("--dereverb-backing", action="store_true",
                    help="Also run dereverb (and de-echo, if on) on the "
                         "backing stem. Output becomes BACKING_DRY.wav.")
    ap.add_argument("--no-karaoke", action="store_true",
                    help="Skip the lead/backing split (it runs by default with "
                         "the becruily & frazer BS karaoke model). The whole "
                         "vocal track becomes LEAD; no BACKING.wav.")
    ap.add_argument("--karaoke-model", type=str, default=None,
                    metavar="MODEL",
                    help="Run the lead/backing split with a different model. "
                         "Shortcuts: 'frazer' (default), 'anvuew', 'becruily', "
                         "'gabox', 'gabox2', 'viperx', '5hp', '6hp', 'mdx', "
                         "'mdx2'. Or a full model filename.")
    ap.add_argument("--aggression", type=int, default=5, metavar="N",
                    help="VR Arch aggression (default 5). Higher = more aggressive "
                         "stem removal. Only affects VR Arch models ('vr' "
                         "dereverb, 'vr-*' de-echo, 5hp/6hp karaoke) — the "
                         "default chain is all Roformers and ignores this.")
    ap.add_argument("--no-dereverb", action="store_true",
                    help="Skip the dereverb pass. Use when dereverb is destroying "
                         "your target vocal (dereverb models can treat rapped "
                         "vocals as 'reverb content').")
    ap.add_argument("--dereverb-model", type=str, default=None,
                    metavar="MODEL",
                    help="Override the dereverb model. Shortcuts: 'anvuew' "
                         "(the default, BS stereo), 'anvuew-mel', 'anvuew-less', "
                         "'mono', 'big', 'super-big', 'echo', 'echo-fused', "
                         "'vr', 'bs'. Or a full model filename.")
    ap.add_argument("--dereverb-passes", type=int, default=1, metavar="N",
                    help="Run the dereverb model N times in sequence (default 1). "
                         "Stack 2-3 passes for heavy reverb. Diminishing returns "
                         "after 3, risk of artifacts.")
    ap.add_argument("--deecho", action="store_true",
                    help="Add stage 4: strip echo/delay from the lead after "
                         "dereverb (Sucial MelBand de-reverb-echo V2).")
    ap.add_argument("--deecho-model", type=str, default=None, metavar="MODEL",
                    help="De-echo with a different model (implies --deecho). "
                         "Shortcuts: 'sucial' (default), 'sucial-fused' (big "
                         "reverb tails), 'vr-normal', 'vr-aggressive' (heavy "
                         "delay; output tops out ~17.5 kHz). Or a full model "
                         "filename.")
    ap.add_argument("--overlap", type=int, default=16, metavar="N",
                    help="Overlapping prediction windows per Roformer pass "
                         "(default 16). Pass time scales linearly with N; "
                         "4 measured within 0.01 dB SDR of 16 at 1/4 the "
                         "time.")
    ap.add_argument("--music", action="store_true",
                    help="Mix the instrumental and backing vocals together into "
                         "a single MUSIC.mp3. Requires ffmpeg in PATH.")
    args = ap.parse_args()

    if args.file:
        song = Path(args.file)
        if not song.is_file():
            print(f"File not found: {song}", file=sys.stderr)
            return 1
    else:
        query = " ".join(args.query)
        if not query:
            ap.print_help()
            return 1

        if not MUSIC_DIR.is_dir():
            print(f"Music library not found: {MUSIC_DIR}\n"
                  f"Set HEXPREP_MUSIC_DIR to your music folder, or use --file.",
                  file=sys.stderr)
            return 1

        matches = find_songs(query)
        if args.list:
            for p in matches:
                print(p)
            print(f"\n{len(matches)} match(es)")
            return 0

        song = pick_song(matches)
        if song is None:
            return 1

    def resolve(name: str | None, shortcuts: dict[str, str]) -> str | None:
        return shortcuts.get(name, name) if name else None

    try:
        process(song, keep_intermediate=args.keep_intermediate,
                swap_lead=args.swap_lead,
                dereverb_backing=args.dereverb_backing,
                no_karaoke=args.no_karaoke,
                karaoke_model_override=resolve(args.karaoke_model, KARAOKE_SHORTCUTS),
                vr_aggression=args.aggression,
                no_dereverb=args.no_dereverb,
                mix_music=args.music,
                dereverb_model_override=resolve(args.dereverb_model, DEREVERB_SHORTCUTS),
                dereverb_passes=args.dereverb_passes,
                vocal_model_override=resolve(args.vocal_model, VOCAL_SHORTCUTS),
                deecho=args.deecho,
                deecho_model_override=resolve(args.deecho_model, DEECHO_SHORTCUTS),
                overlap=args.overlap)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
