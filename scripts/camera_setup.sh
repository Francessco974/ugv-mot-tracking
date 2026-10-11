#!/usr/bin/env bash
# camera_setup.sh — put the USB camera in a known, verified state.
# Usage: ./camera_setup.sh [MJPG|YUYV] [WIDTH] [HEIGHT] [FPS]
# Override the device with CAMERA_DEV=/dev/v4l/by-id/... if needed.
set -euo pipefail

FMT=${1:-MJPG}; W=${2:-1280}; H=${3:-720}; FPS=${4:-30}
DEV=${CAMERA_DEV:-$(ls /dev/v4l/by-id/*video-index0 2>/dev/null | head -n1 || true)}

if [ -z "$DEV" ] || [ ! -e "$DEV" ]; then
  echo "ERROR: no camera in /dev/v4l/by-id/ (set CAMERA_DEV)" >&2; exit 1
fi
echo "Device: $DEV -> $(readlink -f "$DEV")"

# ---- set ----
v4l2-ctl -d "$DEV" --set-fmt-video=width="$W",height="$H",pixelformat="$FMT"
v4l2-ctl -d "$DEV" --set-parm="$FPS" > /dev/null
v4l2-ctl -d "$DEV" --set-ctrl=exposure_dynamic_framerate=0

# ---- verify ----
fail=0
check() {
  if [ "$2" = "$3" ]; then echo "ok    $1 = $2"
  else echo "FAIL  $1: wanted $3, got '$2'" >&2; fail=1; fi
}
fmt=$(v4l2-ctl -d "$DEV" --get-fmt-video)
check width  "$(sed -n 's|.*Width/Height *: *\([0-9]*\)/.*|\1|p' <<< "$fmt")" "$W"
check height "$(sed -n 's|.*Width/Height *: *[0-9]*/\([0-9]*\).*|\1|p' <<< "$fmt")" "$H"
check format "$(sed -n "s|.*Pixel Format *: *'\([A-Z0-9]*\)'.*|\1|p" <<< "$fmt")" "$FMT"
check fps    "$(v4l2-ctl -d "$DEV" --get-parm | sed -n 's|.*Frames per second: *\([0-9]*\)\..*|\1|p')" "$FPS"
check exposure_dynamic_framerate \
      "$(v4l2-ctl -d "$DEV" --get-ctrl=exposure_dynamic_framerate | awk '{print $2}')" "0"
exit $fail