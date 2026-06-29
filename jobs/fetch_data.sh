#!/usr/bin/env bash
# ================================================================
# Jet job: download all eval datasets (Flickr30k, MSCOCO).
# Needs internet access. CPU pod is sufficient.
# Run this BEFORE the H100 training job.
# ================================================================
PROJECT_ROOT="/media/beegfs/users/suyash.b/projects/my_projects/vlm_distillation"
VENV_PATH="$PROJECT_ROOT/.venv"
JOB_NAME="vlm-distil-fetch-data"

COMMAND="python data/download_vqa.py --config configs/distil_config.yaml --workers 32"

echo "🚀 Launching Jet job"
echo "   Job name : $JOB_NAME"
echo "   Project  : $PROJECT_ROOT"
echo "   Command  : $COMMAND"

jet launch job "$JOB_NAME" \
  --image registry.gitlab.com/hvlabs/teams/ai/container-images/base:ubuntu24.04-cuda13.0.2-runtime-withtools-v1.0.1 \
  --image-pull-secrets hv-gitlab-registry \
  --scheduler kai-scheduler \
  --image-pull-policy IfNotPresent \
  --cpu 32:40 \
  --memory 100Gi:180Gi \
  --shm-size 8Gi \
  --volume /media/beegfs:/media/beegfs \
  --working-dir "$PROJECT_ROOT" \
  --pyenv "$VENV_PATH" \
  --command "$COMMAND" \
  --follow
