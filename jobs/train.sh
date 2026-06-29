#!/bin/bash
# ================================================================
# CONFIG
# ================================================================
PROJECT_ROOT="/media/beegfs/users/suyash.b/projects/my_projects/vlm_distillation"
VENV_PATH="$PROJECT_ROOT/.venv"
JOB_NAME="vlm-distil-train"

COMMAND="bash -c 'export HF_HUB_OFFLINE=1 && python training/train.py --config configs/distil_config.yaml'"

echo "Launching: $JOB_NAME"
echo "  Project : $PROJECT_ROOT"
echo "  Command : $COMMAND"

# ================================================================
# LAUNCH
# ================================================================
jet launch job "$JOB_NAME" \
  --image registry.gitlab.com/hvlabs/teams/ai/container-images/base:ubuntu24.04-cuda13.0.2-runtime-withtools-v1.0.1 \
  --image-pull-secrets hv-gitlab-registry \
  --scheduler kai-scheduler \
  --image-pull-policy IfNotPresent \
  --gpu 1 \
  --gpu-type h100 \
  --cpu 8:32 \
  --shm-size 16Gi \
  --memory 32Gi:128Gi \
  --volume /media/beegfs:/media/beegfs \
  --working-dir "$PROJECT_ROOT" \
  --pyenv "$VENV_PATH" \
  --command "$COMMAND" \
  --follow
