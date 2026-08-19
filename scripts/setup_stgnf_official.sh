#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/orhir/STG-NF.git"
PINNED_COMMIT="edb5f3220332e160e4d20ce258787d5e2d7e0200"
DEST="${1:-third_party/stgnf_official}"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO_URL" "$DEST"
fi

git -C "$DEST" fetch --all --tags --prune
git -C "$DEST" checkout --detach "$PINNED_COMMIT"

ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
if [ "$ACTUAL" != "$PINNED_COMMIT" ]; then
  echo "ERROR: expected $PINNED_COMMIT, got $ACTUAL" >&2
  exit 1
fi

echo "STG-NF official repo ready"
echo "commit: $ACTUAL"
