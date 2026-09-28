#!/usr/bin/env bash
# Copy ONNX Runtime Web into lib/ort/, so it is served from the experiment's own origin.
#
# Needed for the multi-threaded CPU encoder. A cross-origin-isolated page gives ONNX Runtime
# SharedArrayBuffer, and with it threads, but ORT starts each thread as a Worker from the URL its
# own script came from, and a page may only start workers from its own origin. Loaded from
# jsDelivr, the threads never start (and once the page is isolated, ORT tries to use them by
# default, on the main thread too). Served from lib/ort/, they do.
#
# The version must match the one saccade.js is built against (ORT_VERSION in lib/saccade/core.js).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/lib/ort"
VERSION="${1:-$(grep -oE 'const ve="[0-9.]+"' "$HERE/lib/saccade/core.js" | grep -oE '[0-9.]+' || true)}"
if [ -z "$VERSION" ]; then
  echo "Could not read ORT's version from lib/saccade/core.js; pass it: scripts/sync-ort.sh 1.29.0" >&2
  exit 1
fi
BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@$VERSION/dist"

mkdir -p "$DEST"
echo "ONNX Runtime Web $VERSION -> lib/ort/"
# The bundle, and the one wasm build it loads: the "jsep" build covers both the WebGPU and the
# wasm (CPU) execution providers.
for f in ort.bundle.min.mjs ort-wasm-simd-threaded.jsep.mjs ort-wasm-simd-threaded.jsep.wasm; do
  curl -sfL "$BASE/$f" -o "$DEST/$f"
  echo "  $f  $(du -h "$DEST/$f" | cut -f1)"
done
printf '{ "onnxruntime-web": "%s", "synced": "%s" }\n' "$VERSION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$DEST/SYNCED.json"
