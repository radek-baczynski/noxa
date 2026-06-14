#!/usr/bin/env bash
# Detect host architecture / GPU backend and start Noxa with an optimized profile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE_OVERRIDE=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/noxa.sh [--profile PROFILE] [--] [uvicorn args...]

Detects the best NOXA_RUNTIME_PROFILE for this machine and starts Noxa.

Profiles:
  mac-local   Apple Silicon (Metal, all GPU layers)
  cloud-gpu   NVIDIA CUDA (all GPU layers)
  cloud-cpu   CPU-only inference

Options:
  --profile PROFILE   Skip auto-detection and use this profile
  --help              Show this help

Environment:
  Loads .env from the repo root. Auto-detected profile overrides
  NOXA_RUNTIME_PROFILE from .env unless --profile is passed.

Examples:
  ./scripts/noxa.sh
  ./scripts/noxa.sh --profile cloud-cpu
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "error: --profile requires a value" >&2; exit 1; }
      PROFILE_OVERRIDE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "$ROOT/.env"
  set +a
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found; install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

detect_runtime() {
  uv run python - <<'PY'
import contextlib
import os
import platform
import sys

try:
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        import llama_cpp
        info = llama_cpp.llama_print_system_info().decode("utf-8", errors="replace")
except ImportError:
    print("error: llama-cpp-python not installed; run: uv sync --extra ml", file=sys.stderr)
    sys.exit(1)

has_cuda = "CUDA" in info
has_metal = "Metal" in info or "MTL" in info

os_name = platform.system()
machine = platform.machine().lower()

if os_name == "Darwin" and machine in {"arm64", "aarch64"}:
    profile = "mac-local"
    device = "metal (all layers)" if has_metal else "cpu"
elif has_cuda:
    profile = "cloud-gpu"
    device = "cuda (all layers)"
else:
    profile = "cloud-cpu"
    device = "cpu (no GPU offload)"

print(f"{profile}\t{device}\t{os_name}/{machine}")
PY
}

if [[ -n "$PROFILE_OVERRIDE" ]]; then
  case "$PROFILE_OVERRIDE" in
    mac-local|cloud-gpu|cloud-cpu) PROFILE="$PROFILE_OVERRIDE" ;;
    *)
      echo "error: unknown profile '$PROFILE_OVERRIDE' (use mac-local, cloud-gpu, or cloud-cpu)" >&2
      exit 1
      ;;
  esac
  OS_ARCH="$(uname -s)/$(uname -m)"
  case "$PROFILE" in
    mac-local) DEVICE="metal (all layers)" ;;
    cloud-gpu) DEVICE="cuda (all layers)" ;;
    cloud-cpu) DEVICE="cpu (no GPU offload)" ;;
  esac
else
  IFS=$'\t' read -r PROFILE DEVICE OS_ARCH < <(detect_runtime)
fi

export NOXA_RUNTIME_PROFILE="$PROFILE"

echo "Noxa startup"
echo "  host:    $OS_ARCH"
echo "  profile: $NOXA_RUNTIME_PROFILE"
echo "  device:  $DEVICE"
if ((${#EXTRA_ARGS[@]} > 0)); then
  echo "  cmd:     uv run noxa ${EXTRA_ARGS[*]}"
  exec uv run noxa "${EXTRA_ARGS[@]}"
else
  echo "  cmd:     uv run noxa"
  exec uv run noxa
fi
