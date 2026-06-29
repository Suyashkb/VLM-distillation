"""
Entry point for distillation training.

Usage:
    python training/train.py --config configs/distil_config.yaml
    python training/train.py --config configs/distil_config.yaml --no_teacher_cache
"""
import os 
import sys 

os.environ.setdefault("USER", "suyash.b")

_BASE_CACHE = "/media/beegfs/users/suyash.b/.cache"
os.makedirs(_BASE_CACHE, exist_ok=True)

for _key, _rel in [
    ("HF_HOME",                 "huggingface"),
    ("HUGGINGFACE_HUB_CACHE",   "huggingface/hub"),
    ("TRANSFORMERS_CACHE",      "huggingface/transformers"),
    ("TORCH_HOME",              "torch"),
    ("MPLCONFIGDIR",            "matplotlib"),
    ("TORCHINDUCTOR_CACHE_DIR", "torch/inductor"),
    ("TRITON_CACHE_DIR",        "torch/triton"),
]:
    _path = os.path.join(_BASE_CACHE, _rel)
    os.makedirs(_path, exist_ok=True)
    os.environ.setdefault(_key, _path)

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vqa_dataset import build_dataloader
from models.distillation_losses import DistillationLoss
from models.student import build_student
from models.teacher import cache_teacher_features, load_teacher
from training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distil_config.yaml")
    parser.add_argument(
        "--no_teacher_cache",
        action="store_true",
        help="Run teacher live each step instead of using cached features (slower)",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run 10 steps then exit")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load answer vocab to get num_answers
    vocab_file = Path(cfg["data"]["cache_dir"]) / "answer_vocab.json"
    if not vocab_file.exists():
        print(f"ERROR: Run 'python data/download_vqa.py' first to build {vocab_file}")
        sys.exit(1)
    with open(vocab_file) as f:
        vocab = json.load(f)
    num_answers = len(vocab)
    print(f"Answer vocab size: {num_answers}")

    # Smoke test: build student, run 10 steps with no teacher (cache skipped), exit.
    # Must come before teacher caching so `--smoke_test` stays fast (~2 min).
    if args.smoke_test:
        print("\n[Smoke test] Running 10 steps (no teacher, no cache) ...")
        student = build_student(cfg, num_answers).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        loss_fn = DistillationLoss()
        smoke_loader = build_dataloader(
            samples_file=str(Path(cfg["data"]["cache_dir"]) / "vqa_samples.json"),
            vocab_file=str(vocab_file),
            image_dir=str(Path(cfg["data"]["data_dir"]) / "val2014"),
            processor_name=cfg["teacher"]["model_name"],
            teacher_cache_file=None,
            batch_size=4,  # tiny batch — just verifying code paths, not training
            num_workers=0,
            shuffle=False,
        )
        batch = next(iter(smoke_loader))
        for step in range(10):
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            ans = batch["answer_id"].to(device)
            optimizer.zero_grad()
            s_logits, s_feats, _ = student(pv, ii, am)
            loss, metrics = loss_fn(s_logits, ans, s_feats, [])
            loss.backward()
            optimizer.step()
            print(f"  step {step+1}: loss={metrics['loss']:.4f} l_task={metrics['l_task']:.4f}")
        print("Smoke test passed.")
        return

    cache_file = cfg["teacher"]["cache_file"]
    use_cache = not args.no_teacher_cache

    base_loader_kwargs = dict(
        samples_file=str(Path(cfg["data"]["cache_dir"]) / "vqa_samples.json"),
        vocab_file=str(vocab_file),
        image_dir=str(Path(cfg["data"]["data_dir"]) / "val2014"),
        processor_name=cfg["teacher"]["model_name"],
        num_workers=cfg["data"]["num_workers"],
    )

    # Build teacher cache FIRST so dataloaders can see it on disk when they open the h5 file
    teacher = None
    if use_cache and not Path(cache_file).exists():
        print("\nBuilding teacher feature cache (one-time cost, ~5-10 min on H100) ...")
        teacher = load_teacher(cfg, num_answers, device)
        cache_loader = build_dataloader(
            **{**base_loader_kwargs, "teacher_cache_file": None},
            batch_size=cfg["teacher"].get("cache_batch_size", 64),
            shuffle=False,
            drop_last=False,  # must cover every sample so training never hits a missing key
        )
        cache_teacher_features(teacher, cache_loader, cache_file, device)
        del teacher
        torch.cuda.empty_cache()
        teacher = None
    elif not use_cache:
        print("\nLoading teacher for live inference ...")
        teacher = load_teacher(cfg, num_answers, device)

    # Build training dataloaders AFTER caching so the h5 file exists when datasets open it
    common_kwargs = {
        **base_loader_kwargs,
        "teacher_cache_file": cache_file if use_cache else None,
    }
    train_loader = build_dataloader(
        **common_kwargs, batch_size=cfg["training"]["batch_size"], shuffle=True,
    )
    val_loader = build_dataloader(
        **common_kwargs, batch_size=cfg["training"]["batch_size"], shuffle=False,
    )

    student = build_student(cfg, num_answers)
    print(f"Student parameters: {student.count_parameters():,} ({student.count_parameters()/1e6:.1f}M)")

    # Train
    trainer = Trainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
