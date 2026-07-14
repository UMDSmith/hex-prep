"""
hex-prep CLI — multi-pass vocal extraction for voice conversion / karaoke stems.

Pass 1:  UVR-MDX-NET Inst HQ 3   -> split vocals from instrumental
Pass 1b: Kim FT (unwa) roformer  -> strip instrumental bleed from the vocals
Pass 2:  (opt-in) becruily MelBand karaoke -> split lead from backing vocals
Pass 3:  UVR-DeEcho-DeReverb     -> dry the vocal

Default is three passes (isolate + polish + dereverb). The lead/backing
karaoke split runs with --karaoke (becruily model) or --karaoke-model X —
most tracks don't need it, so it stays opt-in.

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
    hex-prep --file /path/to/song.mp3
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from audio_separator.separator import Separator

# ----- Configuration -------------------------------------------------------

MUSIC_DIR = Path(os.getenv("HEXPREP_MUSIC_DIR", str(Path.home() / "Music")))
OUTPUT_DIR = Path(os.getenv("HEXPREP_OUTPUT_DIR", str(Path.home() / "hex-prep-output")))
MODEL_DIR = Path(os.getenv("HEXPREP_MODEL_DIR", str(Path.home() / "audio-separator-models")))
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".wma"}

# Single chain, picked by A/B listening tests in UVR5 plus measured checks:
#   isolate  — Inst HQ 3: best vocal/music split tested; instrumental-focused,
#              so vocals are the inverse and rap survives by construction.
#   polish   — kim_ft_unwa (vocal-focused roformer) re-separates the vocal
#              stem to strip instrumental bleed HQ3 leaves behind (sparse
#              acoustic guitar etc. — measured -37.7dB -> -91dB on the
#              guitar-only intro of Tracy Chapman's Fast Car). The
#              instrumental stem still comes from HQ3.
#   karaoke  — becruily MelBand karaoke: best open lead/backing model that
#              audio-separator supports (MVSEP lead/back leaderboard top open
#              entry; verified on Bohemian Rhapsody: harmony intro -> backing,
#              solo verse -> lead). Opt-in via --karaoke / --karaoke-model —
#              most tracks don't need the split.
#   dereverb — UVR-DeEcho-DeReverb (VR Arch): best dereverb tested.
#              Responds to --aggression.
MODELS = {
    "isolate":  "UVR-MDX-NET-Inst_HQ_3.onnx",
    "polish":   "mel_band_roformer_kim_ft_unwa.ckpt",
    "karaoke":  "mel_band_roformer_karaoke_becruily.ckpt",
    "dereverb": "UVR-DeEcho-DeReverb.pth",
}

KARAOKE_SHORTCUTS = {
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
    "anvuew":      "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "anvuew-less": "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    "mono":        "dereverb_mel_band_roformer_mono_anvuew.ckpt",
    "big":         "dereverb_big_mbr_ep_362.ckpt",
    "super-big":   "dereverb_super_big_mbr_ep_346.ckpt",
    "echo":        "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    "echo-fused":  "dereverb_echo_mbr_fused.ckpt",
    "vr":          "UVR-DeEcho-DeReverb.pth",
    "bs":          "deverb_bs_roformer_8_384dim_10depth.ckpt",
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

def make_separator(work_dir: Path, vr_aggression: int = 5) -> Separator:
    """One Separator instance reused across all passes — model swap is
    cheap, separator init is not. vr_aggression controls how hard VR Arch
    models push to remove the target stem (5 default, 10-15 for cleaner
    karaoke at risk of artifacts)."""
    return Separator(
        output_dir=str(work_dir),
        model_file_dir=str(MODEL_DIR),
        output_format="WAV",
        sample_rate=44100,
        vr_params={"aggression": vr_aggression},
    )


def run_pass(sep: Separator, label: str, model: str, input_path: Path) -> list[Path]:
    """Load `model`, separate `input_path`, return the resulting output paths."""
    print(f"\n--- Pass: {label} ---")
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


def pick_stem(paths: list[Path], wanted: str) -> Path:
    """audio-separator names outputs with the stem type in parentheses, e.g.
    'foo_(Vocals)_BS-Roformer.wav'. After multiple passes, names accumulate
    multiple parenthesized tags from prior passes, e.g.
    'foo_(Vocals)_VocFT_(Instrumental)_Karaoke.wav' — so we must match the
    LAST parenthesized tag (the one this pass produced), not just any
    substring match. Otherwise the lead/backing assignment can flip
    silently when filenames share earlier tags."""
    import re
    wanted_lower = wanted.lower()
    paren_re = re.compile(r"\(([^)]+)\)")

    for p in paths:
        tags = paren_re.findall(p.name.lower())
        if tags and tags[-1] == wanted_lower:
            return p

    # Fallback 1: any parenthesized tag matches
    for p in paths:
        tags = paren_re.findall(p.name.lower())
        if wanted_lower in tags:
            return p

    # Fallback 2: substring match without parens (last resort)
    for p in paths:
        if wanted_lower in p.name.lower():
            return p

    raise RuntimeError(
        f"Couldn't find a '{wanted}' stem in outputs: {[p.name for p in paths]}"
    )


# ----- Main pipeline -------------------------------------------------------

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
            karaoke: bool = False,
            karaoke_model_override: str | None = None,
            vr_aggression: int = 5,
            no_dereverb: bool = False,
            no_polish: bool = False,
            mix_music: bool = False,
            dereverb_model_override: str | None = None,
            dereverb_passes: int = 1) -> dict[str, Path]:
    """Run the extraction chain on `song`. Returns final output paths keyed
    by stem name ("lead", "instrumental", "backing", "backing_dry", "music" —
    only keys that were actually produced)."""
    models = dict(MODELS)
    chain_label = "HQ3+KimFT+DeEcho"
    if no_polish:
        chain_label = "HQ3+DeEcho / NO-POLISH"

    # Karaoke split only runs when asked for (--karaoke uses the default
    # becruily model; --karaoke-model swaps it and implies --karaoke)
    no_karaoke = not karaoke and karaoke_model_override is None
    if karaoke_model_override:
        models["karaoke"] = karaoke_model_override
    if not no_karaoke:
        chain_label += f" / KARAOKE={models['karaoke'][:30]}"

    # If user supplied a custom dereverb model, override and force dereverb on
    if dereverb_model_override:
        models["dereverb"] = dereverb_model_override
        no_dereverb = False
        chain_label += f" / DEREVERB={dereverb_model_override[:30]}"

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = OUTPUT_DIR / song.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Hex Prep ({chain_label})")
    print(f"  Source: {song}")
    print(f"  Output: {work_dir}")
    print(f"{'='*60}")

    sep = make_separator(work_dir, vr_aggression=vr_aggression)
    pipeline_start = time.time()

    # Pass 1: isolate vocals
    p1_outputs = run_pass(sep, "Vocal isolation", models["isolate"], song)
    vocals_stem = pick_stem(p1_outputs, "Vocals")
    instrumental_stem = pick_stem(p1_outputs, "Instrumental")

    # Pass 1b: polish — re-separate the vocal stem with a vocal-focused model
    # to strip instrumental bleed HQ3 left in it. The polish model's own
    # "Instrumental" output (the removed bleed) is discarded; the music stem
    # stays HQ3's.
    if not no_polish:
        p1b_outputs = run_pass(sep, "Bleed polish", models["polish"], vocals_stem)
        vocals_stem = pick_stem(p1b_outputs, "Vocals")

    # Pass 2: split lead from backing (skipped unless --karaoke)
    if no_karaoke:
        print("\n--- Skipping karaoke split (enable with --karaoke) ---")
        lead_stem = vocals_stem      # entire vocal track becomes "lead"
        backing_stem = None
    else:
        p2_outputs = run_pass(sep, "Lead/backing split", models["karaoke"], vocals_stem)
        # In practice across the karaoke models we use (5hp, 6hp, MDX-Net Karaoke,
        # Mel-Roformer karaoke), the "(Vocals)" output stem is the lead vocal and
        # the "(Instrumental)" stem is the backing/harmony content. Despite some
        # model docs claiming the opposite naming convention, empirical testing
        # on real songs consistently shows this is the right assignment.
        # --swap-lead inverts this for the rare model/track where it's wrong.
        if swap_lead:
            lead_stem = pick_stem(p2_outputs, "Instrumental")
            backing_stem = pick_stem(p2_outputs, "Vocals")
        else:
            lead_stem = pick_stem(p2_outputs, "Vocals")
            backing_stem = pick_stem(p2_outputs, "Instrumental")

    # Pass 3: dereverb the lead (skipped if --no-dereverb)
    def _extract_dry(outputs: list[Path]) -> Path:
        """Pick the dry/no-reverb stem from dereverb model outputs. Like
        pick_stem, must match the LAST parenthesized tag — after multiple
        passes, filenames accumulate prior tags ('noreverb' from earlier
        passes appears in both the dry AND wet outputs of the next pass)."""
        import re
        paren_re = re.compile(r"\(([^)]+)\)")
        dry_tags = {"noreverb", "no reverb", "no_reverb", "dry"}

        for p in outputs:
            tags = paren_re.findall(p.name.lower())
            if tags and tags[-1] in dry_tags:
                return p

        # Fallback: any parenthesized tag matches a dry indicator
        for p in outputs:
            tags = paren_re.findall(p.name.lower())
            if any(t in dry_tags for t in tags):
                return p

        print(f"  WARN: couldn't auto-detect dry stem, defaulting to {outputs[0].name}")
        return outputs[0]

    if no_dereverb:
        print("\n--- Skipping dereverb (--no-dereverb) ---")
        dry_lead = lead_stem
    else:
        # Dereverb pass(es). Stacking multiple passes is the most effective way
        # to handle heavy reverb (e.g. cathedral-style verb) since the
        # Mel-Roformer dereverb models don't expose aggressiveness.
        # We rename intermediate outputs to short canonical names because
        # audio-separator appends model names to filenames every pass, which
        # quickly exceeds the 255-byte filename limit on stacked passes.
        current_input = lead_stem
        dry_lead = lead_stem
        for pass_num in range(1, dereverb_passes + 1):
            label = (f"Dereverb lead (pass {pass_num}/{dereverb_passes})"
                     if dereverb_passes > 1 else "Dereverb lead")
            p3_outputs = run_pass(sep, label, models["dereverb"], current_input)
            picked_dry = _extract_dry(p3_outputs)

            # If more passes coming, rename to a short canonical name so the
            # filename doesn't grow unbounded. Last pass keeps its full name
            # since we'll rename it to __LEAD_DRY.wav anyway.
            if pass_num < dereverb_passes:
                short_name = work_dir / f"_dereverb_intermediate_pass{pass_num}.wav"
                if short_name.exists():
                    short_name.unlink()
                picked_dry.rename(short_name)
                current_input = short_name
                dry_lead = short_name
            else:
                dry_lead = picked_dry

    # Optional Pass 3b: dereverb the backing too (useful when the karaoke model
    # split the wrong vocal into backing, or you want both stems clean for
    # layering)
    dry_backing = None
    if dereverb_backing and backing_stem is not None and not no_dereverb:
        p3b_outputs = run_pass(sep, "Dereverb backing",
                                models["dereverb"], backing_stem)
        dry_backing = _extract_dry(p3b_outputs)

    # Rename final outputs to predictable names
    lead_suffix = "LEAD" if no_dereverb else "LEAD_DRY"
    final_lead = work_dir / f"{song.stem}__{lead_suffix}.wav"
    final_inst = work_dir / f"{song.stem}__INSTRUMENTAL.wav"
    final_backing = work_dir / f"{song.stem}__BACKING.wav"
    final_backing_dry = work_dir / f"{song.stem}__BACKING_DRY.wav"

    # Use copy if dry_lead and lead_stem are the same file (no_dereverb mode);
    # otherwise rename.
    if dry_lead == lead_stem:
        shutil.copy2(dry_lead, final_lead)
    else:
        dry_lead.rename(final_lead)
    instrumental_stem.rename(final_inst)
    if backing_stem is not None:
        if dry_backing is not None:
            # We have a dereverbed version — that becomes BACKING_DRY,
            # original wet backing gets cleaned up unless keeping intermediates
            dry_backing.rename(final_backing_dry)
        else:
            backing_stem.rename(final_backing)

    # Optional: mix instrumental + backing into a single music file (for
    # workflows where the vocal stem drives something like lip-sync and one
    # music file plays everything else). Runs BEFORE cleanup so the music
    # file gets preserved even with intermediate cleanup.
    final_music = None
    if mix_music:
        final_music = mix_to_music(
            instrumental=final_inst,
            backing=final_backing if (backing_stem is not None and dry_backing is None) else
                    final_backing_dry if dry_backing is not None else None,
            output=work_dir / f"{song.stem}__MUSIC.mp3",
        )

    # Cleanup intermediates unless requested otherwise
    if not keep_intermediate:
        keep = {final_lead, final_inst, final_backing, final_backing_dry}
        if final_music is not None:
            keep.add(final_music)
        for p in work_dir.glob("*.wav"):
            if p not in keep:
                p.unlink()

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Done in {total:.1f}s")
    print(f"  Lead vocal:        {final_lead}")
    print(f"  Instrumental:      {final_inst}")
    if dry_backing is not None:
        print(f"  Backing (dry):     {final_backing_dry}")
    elif backing_stem is not None:
        print(f"  Backing vocals:    {final_backing}")
    if final_music is not None:
        print(f"  Music mix (.mp3):  {final_music}")
    print(f"{'='*60}\n")

    results: dict[str, Path] = {"lead": final_lead, "instrumental": final_inst}
    if dry_backing is not None:
        results["backing_dry"] = final_backing_dry
    elif backing_stem is not None:
        results["backing"] = final_backing
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
    ap.add_argument("--swap-lead", action="store_true",
                    help="Invert lead/backing assignment (only meaningful with "
                         "--karaoke; use when the model picks the wrong vocal "
                         "as lead)")
    ap.add_argument("--dereverb-backing", action="store_true",
                    help="Also dereverb the backing stem (only meaningful with "
                         "--karaoke). Output becomes BACKING_DRY.wav.")
    ap.add_argument("--karaoke", action="store_true",
                    help="Run the lead/backing split with the default model "
                         "(becruily MelBand karaoke — best open lead/back model "
                         "audio-separator supports). Adds BACKING.wav; LEAD "
                         "becomes just the lead vocal.")
    ap.add_argument("--karaoke-model", type=str, default=None,
                    metavar="MODEL",
                    help="Run the lead/backing split with a different model. "
                         "Shortcuts: 'becruily' (default), 'gabox', 'gabox2', "
                         "'viperx', '5hp', '6hp', 'mdx', 'mdx2'. Or a full model "
                         "filename. (Models download automatically.)")
    ap.add_argument("--aggression", type=int, default=5, metavar="N",
                    help="VR Arch aggression (default 5). Higher = more aggressive "
                         "stem removal. Affects the default DeEcho-DeReverb pass "
                         "and VR Arch karaoke models (5hp/6hp).")
    ap.add_argument("--no-dereverb", action="store_true",
                    help="Skip the dereverb pass. Use when dereverb is destroying "
                         "your target vocal (dereverb models can treat rapped "
                         "vocals as 'reverb content').")
    ap.add_argument("--no-polish", action="store_true",
                    help="Skip the bleed-polish pass (Kim FT re-separation of "
                         "the vocal stem). Use if polish ever eats part of the "
                         "vocal on a specific track.")
    ap.add_argument("--dereverb-model", type=str, default=None,
                    metavar="MODEL",
                    help="Override the dereverb model (default is VR Arch "
                         "DeEcho-DeReverb). Shortcuts: 'vr' (the default), "
                         "'anvuew', 'anvuew-less', 'big', 'super-big', 'echo', "
                         "'echo-fused', 'mono', 'bs'. Or a full model filename. "
                         "(Models download automatically.)")
    ap.add_argument("--dereverb-passes", type=int, default=1, metavar="N",
                    help="Run the dereverb model N times in sequence (default 1). "
                         "Stack 2-3 passes for heavy reverb. Diminishing returns "
                         "after 3, risk of artifacts.")
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

    karaoke_override = args.karaoke_model
    if karaoke_override and karaoke_override in KARAOKE_SHORTCUTS:
        karaoke_override = KARAOKE_SHORTCUTS[karaoke_override]

    dereverb_override = args.dereverb_model
    if dereverb_override and dereverb_override in DEREVERB_SHORTCUTS:
        dereverb_override = DEREVERB_SHORTCUTS[dereverb_override]

    try:
        process(song, keep_intermediate=args.keep_intermediate,
                swap_lead=args.swap_lead,
                dereverb_backing=args.dereverb_backing,
                karaoke=args.karaoke,
                karaoke_model_override=karaoke_override,
                vr_aggression=args.aggression,
                no_dereverb=args.no_dereverb,
                no_polish=args.no_polish,
                mix_music=args.music,
                dereverb_model_override=dereverb_override,
                dereverb_passes=args.dereverb_passes)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
