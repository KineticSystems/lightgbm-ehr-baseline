#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
WEIGHT_DIR="$PAPER_ROOT/.export_assets/external_artifacts/torchvision_resnet34_imagenet1k_v1"
WEIGHT_PATH="$WEIGHT_DIR/resnet34-b627a593.pth"
WEIGHT_URL="https://download.pytorch.org/models/resnet34-b627a593.pth"
EXPECTED_SHA="b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"

mkdir -p "$WEIGHT_DIR"

if [[ ! -f "$WEIGHT_PATH" ]]; then
  TEMP_PATH=$(mktemp "$WEIGHT_DIR/.resnet34-download.XXXXXX")
  trap 'rm -f "$TEMP_PATH"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 \
    --output "$TEMP_PATH" "$WEIGHT_URL"
  DOWNLOADED_SHA=$(sha256sum "$TEMP_PATH" | cut -d' ' -f1)
  if [[ "$DOWNLOADED_SHA" != "$EXPECTED_SHA" ]]; then
    echo "ResNet-34 weight SHA-256 prefix mismatch: $DOWNLOADED_SHA" >&2
    exit 1
  fi
  mv "$TEMP_PATH" "$WEIGHT_PATH"
  trap - EXIT
fi

WEIGHT_SHA=$(sha256sum "$WEIGHT_PATH" | cut -d' ' -f1)
if [[ "$WEIGHT_SHA" != "$EXPECTED_SHA" ]]; then
  echo "ResNet-34 weight SHA-256 prefix mismatch: $WEIGHT_SHA" >&2
  exit 1
fi

echo "$WEIGHT_SHA  $WEIGHT_PATH"
