#!/usr/bin/env bash
# ================================================================
# Jet job: run all 4 studies sequentially on H100.
# Total wall time: ~3.5h
# Requires: setup done, data pre-fetched (CC3M + Flickr30k + MSCOCO)
# ================================================================
PROJECT_ROOT="/media/beegfs/users/suyash.b/projects/my_projects/vlm_distillation"
VENV_PATH="$PROJECT_ROOT/.venv"
JOB_NAME="vlm-distil-benchmark-new"

# Train + evaluate + benchmark in sequence
COMMAND="bash -c 'export HF_HUB_OFFLINE=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python training/train.py --config configs/distil_config.yaml && python eval/evaluate.py --teacher --config configs/distil_config.yaml && python eval/evaluate.py --checkpoint checkpoints/best.pt --config configs/distil_config.yaml && python benchmarking/benchmark.py --config configs/distil_config.yaml'"

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
  --cpu 16:32 \
  --memory 64Gi:128Gi \
  --shm-size 48Gi \
  --volume /media/beegfs:/media/beegfs \
  --working-dir "$PROJECT_ROOT" \
  --pyenv "$VENV_PATH" \
  --command "$COMMAND" \
  --follow
