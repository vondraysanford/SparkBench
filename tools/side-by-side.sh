#!/usr/bin/env bash
# tools/side-by-side.sh — two recordings -> one side-by-side GIF + MP4 (frontier vs. fine-tune)
#
# Usage:
#   tools/side-by-side.sh LEFT RIGHT OUTPUT_BASENAME [options]
#
# Options:
#   --height PX   height both panes are scaled to before stacking   (default 720)
#   --fps N       GIF frame rate                                     (default 12)
#   --speed X     playback multiplier applied to both                (default 1)
#
# Record each condition separately at the SAME terminal size, font, and theme, then stack them.
# The clip ends when the shorter recording ends. Put the condition name in the shell prompt or
# window title so the panes label themselves.
#
# Output:
#   OUTPUT_BASENAME.gif          committed
#   <dir>/raw/<basename>.mp4     for LinkedIn/X; raw/ is gitignored
#
# Example:
#   tools/side-by-side.sh raw/frontier.mov raw/finetune.mov \
#     experiments/01-finetune-vs-frontier/evidence/phase-6/2026-10-18-side-by-side-review --speed 1.5
set -euo pipefail

usage() { sed -n '2,22p' "$0"; exit 1; }
[ $# -ge 3 ] || usage
L="$1"; R="$2"; OUT="$3"; shift 3
HEIGHT=720; FPS=12; SPEED=1
while [ $# -gt 0 ]; do
  case "$1" in
    --height) HEIGHT="$2"; shift 2 ;;
    --fps)    FPS="$2";    shift 2 ;;
    --speed)  SPEED="$2";  shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done

command -v ffmpeg >/dev/null || { echo "ffmpeg not found — macOS: brew install ffmpeg" >&2; exit 1; }
for f in "$L" "$R"; do [ -f "$f" ] || { echo "no such file: $f" >&2; exit 1; }; done

OUTDIR="$(dirname "$OUT")"; BASE="$(basename "$OUT")"
mkdir -p "$OUTDIR" "$OUTDIR/raw"
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT

# Scale each pane to the same height (width auto, kept even), speed both, stack left|right.
STACK="[0:v]setpts=PTS/${SPEED},scale=-2:${HEIGHT}:flags=lanczos[l];[1:v]setpts=PTS/${SPEED},scale=-2:${HEIGHT}:flags=lanczos[r];[l][r]hstack=inputs=2:shortest=1"

# GIF, two-pass palette (see tools/gif.sh for why).
ffmpeg -v error -y -i "$L" -i "$R" \
  -filter_complex "${STACK},fps=${FPS},palettegen=stats_mode=diff" "$TMPD/palette.png"
ffmpeg -v error -y -i "$L" -i "$R" -i "$TMPD/palette.png" \
  -filter_complex "${STACK},fps=${FPS}[v];[v][2:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 "${OUT}.gif"

# MP4 for social.
ffmpeg -v error -y -i "$L" -i "$R" \
  -filter_complex "${STACK}[v]" -map "[v]" \
  -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart -an \
  "${OUTDIR}/raw/${BASE}.mp4"

if command -v gifsicle >/dev/null; then gifsicle -O3 --batch "${OUT}.gif" 2>/dev/null || true; fi

echo "wrote:"
ls -lh "${OUT}.gif" "${OUTDIR}/raw/${BASE}.mp4" 2>/dev/null | awk '{printf "  %-8s %s\n", $5, $9}'
