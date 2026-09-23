#!/usr/bin/env bash
# Copy the built saccade.js browser bundles and the eye-embedding model into lib/saccade/.
#
# saccade.js is not on npm yet, so this experiment is served from local builds. Run
# `npm run build` in the saccade.js monorepo first, then run this.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$HERE/../saccadejs}"
DEST="$HERE/lib/saccade"

if [ ! -d "$SRC/packages/core/dist" ]; then
  echo "No build found at $SRC/packages/core/dist" >&2
  echo "Usage: scripts/sync-saccade.sh [path-to-saccadejs-repo]  (after 'npm run build' there)" >&2
  exit 1
fi

mkdir -p "$DEST"

copy() { # <package dir> <output name>
  cp "$SRC/packages/$1/dist/index.browser.min.js" "$DEST/$2.js"
  echo "  $2.js"
}

echo "Copying bundles from $SRC:"
copy core core
copy extension-saccadejs extension
copy plugin-saccadejs-preview plugin-preview
copy plugin-saccadejs-time-sync plugin-time-sync
copy plugin-saccadejs-performance plugin-performance

cp "$SRC/packages/core/models/eye_embedding.onnx" "$DEST/eye_embedding.onnx"
echo "  eye_embedding.onnx ($(du -h "$DEST/eye_embedding.onnx" | cut -f1))"

VERSION=$(node -p "require('$SRC/packages/core/package.json').version" 2>/dev/null || echo unknown)
# The exact source the bundles were built from, so a session can be traced back to it. A dirty
# tree means the bundles contain changes no commit has, which is fine for piloting and not for
# collection.
COMMIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
BRANCH=$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no packages 2>/dev/null)" ]; then
  DIRTY=true
else
  DIRTY=false
fi
cat > "$DEST/SYNCED.json" <<JSON
{"synced": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "core_version": "$VERSION", "commit": "$COMMIT", "branch": "$BRANCH", "dirty": $DIRTY}
JSON
echo "Done. saccade.js core $VERSION at $COMMIT ($BRANCH)$([ "$DIRTY" = true ] && echo ', UNCOMMITTED CHANGES')"
