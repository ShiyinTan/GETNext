#!/usr/bin/env bash
# Create the CPU conda env from environment-cpu.yml (idempotent).
# Prefers micromamba, then mamba, then conda.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ENV_NAME="getnext-cpu"
YML="${ROOT}/environment-cpu.yml"

find_conda() {
  if command -v micromamba >/dev/null 2>&1; then
    echo micromamba
  elif command -v mamba >/dev/null 2>&1; then
    echo mamba
  elif command -v conda >/dev/null 2>&1; then
    echo conda
  else
    echo ""
  fi
}

TOOL="$(find_conda)"
if [ -z "${TOOL}" ]; then
  echo "Need conda, mamba, or micromamba. Install Miniconda/Mambaforge, then re-run." >&2
  echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

env_exists() {
  case "${TOOL}" in
    micromamba) micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}" ;;
    *) ${TOOL} env list | awk '{print $1}' | grep -qx "${ENV_NAME}" ;;
  esac
}

create_or_update() {
  if env_exists; then
    echo "Conda env ${ENV_NAME} exists; updating from ${YML}"
    case "${TOOL}" in
      micromamba) micromamba install -y -n "${ENV_NAME}" -f "${YML}" ;;
      mamba) mamba env update -n "${ENV_NAME}" -f "${YML}" ;;
      conda) conda env update -n "${ENV_NAME}" -f "${YML}" ;;
    esac
  else
    echo "Creating conda env ${ENV_NAME} from ${YML} (${TOOL})"
    case "${TOOL}" in
      micromamba) micromamba create -y -f "${YML}" ;;
      mamba) mamba env create -y -f "${YML}" ;;
      conda) conda env create -f "${YML}" ;;
    esac
  fi
}

run_in_env() {
  case "${TOOL}" in
    micromamba) micromamba run -n "${ENV_NAME}" "$@" ;;
    mamba) mamba run -n "${ENV_NAME}" "$@" ;;
    conda) conda run -n "${ENV_NAME}" --no-banner "$@" ;;
  esac
}

create_or_update

if [ ! -f dataset/NYC/NYC_train.csv ]; then
  mkdir -p dataset
  if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is required to unpack dataset/NYC.zip" >&2
    exit 1
  fi
  unzip -o -q dataset/NYC.zip -d dataset
  rm -rf dataset/__MACOSX || true
fi

run_in_env python - <<'PY'
import sys, torch, numpy, pandas, sklearn, yaml, tqdm, networkx, scipy
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "device", "cpu")
print("numpy", numpy.__version__, "pandas", pandas.__version__, "sklearn", sklearn.__version__)
if "+cpu" not in torch.__version__ and not torch.__version__.endswith("cpu"):
    # Some conda CPU builds report 2.4.1 without +cpu; still require cuda False.
    pass
if torch.cuda.is_available():
    raise SystemExit("CPU env unexpectedly has CUDA")
PY

echo "Conda CPU env ready: ${TOOL} activate ${ENV_NAME}"
echo "  ${TOOL} activate ${ENV_NAME}"
echo "  python causal/train.py --no-cuda ..."
