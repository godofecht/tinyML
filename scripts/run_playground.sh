#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

echo "Configuring playground build in $BUILD_DIR..."
cmake -S "$REPO_ROOT" -B "$BUILD_DIR"

echo "Building PlaygroundServer..."
cmake --build "$BUILD_DIR" --target PlaygroundServer

echo "Launching PlaygroundServer (press Ctrl+C to stop)..."
exec "$BUILD_DIR/bin/PlaygroundServer"
