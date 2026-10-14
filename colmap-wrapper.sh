#!/bin/bash
# COLMAP wrapper — sets library path for pre-built macOS binary
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAMEWORKS="$SCRIPT_DIR/colmap-bin/COLMAP.app/Contents/Frameworks"
export DYLD_FALLBACK_LIBRARY_PATH="$FRAMEWORKS:$DYLD_FALLBACK_LIBRARY_PATH"
exec "$SCRIPT_DIR/colmap-bin/COLMAP.app/Contents/MacOS/colmap" "$@"
