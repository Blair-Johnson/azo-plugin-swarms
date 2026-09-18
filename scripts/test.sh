#!/usr/bin/env bash
set -euo pipefail
: "${AZO_HOST_MANIFEST:?Set AZO_HOST_MANIFEST to the integrated Agent Zoo pixi.toml}"
exec env -u PYTHONPATH -u PYTHONHOME pixi run --manifest-path "$AZO_HOST_MANIFEST" --frozen --no-install python -m pytest -q "$@"
