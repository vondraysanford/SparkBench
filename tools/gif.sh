#!/usr/bin/env bash
# tools/gif.sh — screen recording (.mov/.mp4) -> README-ready GIF + social-ready MP4
#
# Usage:
#   tools/gif.sh INPUT OUTPUT_BASENAME [options]
#
# Options:
#   --start SEC      trim: begin at SEC into the input               (default 0)
#   --duration SEC   trim: keep SEC seconds of input after --start   (default: to the end)
#   --speed X        playback multiplier; 2 = twice as fast          (default 1)
#   --fps N          GIF frame rate; 8 for slow terminal output,
#                    12 for most things, 15 for UI motion            (default 12)
#   --width PX       output width, height follows; 1200 for README,
#                    800 for social crops                            (default 1200)
#   --no-mp4         skip the MP4
#
# Output:
#   OUTPUT_BASENAME.gif            committed to git next to the other evidence
#   <dir>/raw/<basename>.mp4       H.264 for LinkedIn/X; raw/ is gitignored
#
# Example:
#   tools/gif.sh ~/Desktop/rec.mov \
#     experiments/01-finetune-vs-frontier/evidence/phase-1/2026-09-13-smoke-train \
#     --start 4 --duration 30 --speed 2
set -euo pipefail

usage() { sed -n '2,24p' "$0"; exit 1; }
[ $# -ge 2 ] || usage
IN="$1"; OUT="$2"; shift 2
START=0; DURATION=""; SPEED=1; FPS=12; WIDTH=1200; MP4=1
while [ $# -gt 0 ]; do
  case "$1" in
    --start)    START="$2";    shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --speed)    SPEED="$2";    shift 2 ;;
    --fps)      FPS="$2";      shift 2 ;;
    --width)    WIDTH="$2";    shift 2 ;;
    --no-mp4)   MP4=0;         shift   ;;
    -h|--help)  usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done

command -v ffmpeg >/dev/null || { echo "ffmpeg not found — macOS: brew install ffmpeg" >&2; exit 1; }
[ -f "$IN" ] || { echo "no such file: $IN" >&2; exit 1; }

OUTDIR="$(dirname "$OUT")"; BASE="$(basename "$OUT")"
mkdir -p "$OUTDIR" "$OUTDIR/raw"
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT

TRIM=(-ss "$START")
[ -n "$DURATION" ] && TRIM+=(-t "$DURATION")
VF="setpts=PTS/${SPEED},fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

# Pass 1: build a 256-colour palette from the actual frames. Far better than ffmpeg's default palette,
# especially for terminals and dark UIs. stats_mode=diff weights colours that change between frames.
ffmpeg -v error -y "${TRIM[@]}" -i "$IN" -vf "${VF},palettegen=stats_mode=diff" "$TMPD/palette.png"

# Pass 2: encode with that palette. Bayer dithering keeps flat colours flat (no crawling noise on
# terminal backgrounds); diff_mode=rectangle only re-encodes the changed region, which shrinks the file.
ffmpeg -v error -y "${TRIM[@]}" -i "$IN" -i "$TMPD/palette.png" \
  -filter_complex "${VF}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 "${OUT}.gif"

if [ "$MP4" = 1 ]; then
  # yuv420p + faststart is what social players and QuickTime expect. -2 keeps height even.
  ffmpeg -v error -y "${TRIM[@]}" -i "$IN" \
    -vf "setpts=PTS/${SPEED},scale=${WIDTH}:-2:flags=lanczos" \
    -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart -an \
    "${OUTDIR}/raw/${BASE}.mp4"
fi

# Optional lossless squeeze if gifsicle is installed (brew install gifsicle).
if command -v gifsicle >/dev/null; then gifsicle -O3 --batch "${OUT}.gif" 2>/dev/null || true; fi

echo "wrote:"
ls -lh "${OUT}.gif" "${OUTDIR}/raw/${BASE}.mp4" 2>/dev/null | awk '{printf "  %-8s %s\n", $5, $9}'
echo "GitHub renders GIFs well under ~10 MB. LinkedIn and X want the MP4."
