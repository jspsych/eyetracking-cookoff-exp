#!/usr/bin/env bash
# Deploy the experiment to https://cogsciresearch.vassar.edu/eyetracking-cookoff/.
#
#   scripts/deploy-vassar.sh            # show what would change, change nothing
#   scripts/deploy-vassar.sh --go       # copy the files
#
# Copies only what the browser needs (index.html, pursuit-only.html, css/, js/, lib/, test/):
# never analysis/, data/ or docs/. Then checks, over HTTPS, that the page is served with the
# isolation headers and that the version it serves is this checkout's.
#
# One-time server setup (already done if the checks pass): the directory, owned by ec2-user,
# and deploy/eyetracking-cookoff.conf installed in /etc/httpd/conf.d/, then a graceful reload.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${COOKOFF_HOST:-cogsci}"
DIR="/var/www/html/eyetracking-cookoff"
URL="https://cogsciresearch.vassar.edu/eyetracking-cookoff"
VERSION="$(grep -oE 'version: "[0-9.]+"' "$HERE/js/config.js" | head -1 | grep -oE '[0-9.]+')"

[ -f "$HERE/lib/ort/ort.bundle.min.mjs" ] || { echo "lib/ort/ is missing: run scripts/sync-ort.sh" >&2; exit 1; }

DRY="--dry-run"
[ "${1:-}" = "--go" ] && DRY=""

echo "Deploying version $VERSION to $HOST:$DIR ${DRY:+(dry run: add --go to copy)}"
rsync -az --delete $DRY --itemize-changes \
  --include='/index.html' --include='/pursuit-only.html' \
  --include='/css/***' --include='/js/***' --include='/lib/***' --include='/test/***' \
  --exclude='*' \
  "$HERE/" "$HOST:$DIR/"

[ -n "$DRY" ] && exit 0

echo "Checking $URL/ ..."
headers="$(curl -sI "$URL/index.html?nocache=$(date +%s)")"
echo "$headers" | grep -qi '^cross-origin-opener-policy: same-origin' || { echo "MISSING: Cross-Origin-Opener-Policy" >&2; exit 1; }
echo "$headers" | grep -qi '^cross-origin-embedder-policy: require-corp' || { echo "MISSING: Cross-Origin-Embedder-Policy" >&2; exit 1; }
served="$(curl -s "$URL/js/config.js?nocache=$(date +%s)" | grep -oE 'version: "[0-9.]+"' | head -1 | grep -oE '[0-9.]+')"
[ "$served" = "$VERSION" ] || { echo "The server serves version '$served', not $VERSION" >&2; exit 1; }
curl -sI "$URL/lib/ort/ort.bundle.min.mjs" | grep -qi '^content-type: text/javascript' || { echo "lib/ort/*.mjs is not served as JavaScript" >&2; exit 1; }
echo "OK: $URL/ serves $VERSION, cross-origin isolated."
