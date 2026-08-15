#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-third_party/infogcn_official}"
REPO="https://github.com/stnoah1/infogcn.git"
COMMIT="873feaa85160317335a83e04013e0ffa3f63525e"

mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  git clone "$REPO" "$DEST"
fi

git -C "$DEST" fetch --all --tags
git -C "$DEST" checkout "$COMMIT"

echo "InfoGCN official repo ready"
echo "path: $DEST"
echo "commit: $(git -C "$DEST" rev-parse HEAD)"
