#!/usr/bin/env bash
# ================================================================
# Jet job: smoke test — 10 training steps per study with random
# weights and fake data. Verifies all code paths before H100 run.
# Any GPU pod works (even a small one). Takes ~2 min.
# ================================================================
PROJECT_ROOT="/media/beegfs/users/suyash.b/projects/my_projects/vlm_distillation"
VENV_PATH="$PROJECT_ROOT/.venv"
JOB_NAME="vlm-distil-smoke-test"

COMMAND="bash -c 'export HF_HUB_OFFLINE=1 && python training/train.py --config configs/distil_config.yaml --smoke_test'"

echo "🚀 Launching Jet job"
echo "   Job name : $JOB_NAME"
echo "   Project  : $PROJECT_ROOT"
echo "   Command  : $COMMAND"

jet launch job "$JOB_NAME" \
  --image registry.gitlab.com/hvlabs/teams/ai/container-images/base:ubuntu24.04-cuda13.0.2-runtime-withtools-v1.0.1 \
  --image-pull-secrets hv-gitlab-registry \
  --scheduler kai-scheduler \
  --image-pull-policy IfNotPresent \
  --gpu 1 \
  --gpu-type h100 \
  --cpu 4:8 \
  --memory 16Gi:32Gi \
  --shm-size 16Gi \
  --volume /media/beegfs:/media/beegfs \
  --working-dir "$PROJECT_ROOT" \
  --pyenv "$VENV_PATH" \
  --command "$COMMAND" \
  --follow
