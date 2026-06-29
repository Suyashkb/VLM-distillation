#!/usr/bin/env bash
# ================================================================
# Jet job: one-time setup — install all Python dependencies
# Run once before fetch_data or smoke_test.
# CPU pod is sufficient.
# ================================================================
PROJECT_ROOT="/media/beegfs/users/suyash.b/projects/my_projects/vlm_distillation"
VENV_PATH="$PROJECT_ROOT/.venv"
UV="/media/beegfs/users/suyash.b/.local/bin/uv"
JOB_NAME="vlm-distil-setup"

# Creates the venv with uv and installs all deps.
# flash-attn needs CUDA to compile, so this runs in a GPU pod.
# The venv lives on beegfs and is reused by all subsequent jet jobs via --pyenv.
# requirements.txt is installed first (no CUDA needed).
# flash-attn is installed second with --no-build-isolation so it can find torch.
COMMAND="UV_CACHE_DIR=/media/beegfs/users/suyash.b/.cache/uv $UV pip install --python $VENV_PATH/bin/python -r requirements.txt"

echo "🚀 Launching Jet job"
echo "   Job name : $JOB_NAME"
echo "   Project  : $PROJECT_ROOT"
echo "   Venv     : $VENV_PATH"
echo "   Command  : $COMMAND"

jet launch job "$JOB_NAME" \
  --image registry.gitlab.com/hvlabs/teams/ai/container-images/base:ubuntu24.04-cuda13.0.2-runtime-withtools-v1.0.1 \
  --image-pull-secrets hv-gitlab-registry \
  --scheduler kai-scheduler \
  --image-pull-policy IfNotPresent \
  --gpu 1 \
  --gpu-type h100 \
  --cpu 8:16 \
  --memory 32Gi:64Gi \
  --shm-size 8Gi \
  --volume /media/beegfs:/media/beegfs \
  --working-dir "$PROJECT_ROOT" \
  --command "$COMMAND" \
  --follow
