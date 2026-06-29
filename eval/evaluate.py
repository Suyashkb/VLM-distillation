"""
VQA accuracy evaluation on the mini-val subset.

Computes exact-match accuracy (answer in top-1 prediction matches majority-vote GT answer).

Usage:
    python eval/evaluate.py --checkpoint checkpoints/best.pt --config configs/distil_config.yaml
    python eval/evaluate.py --teacher --config configs/distil_config.yaml
"""

import os

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
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.vqa_dataset import build_dataloader
from models.student import build_student
from models.teacher import load_teacher


def train_teacher_head_from_cache(
    teacher,
    cfg: dict,
    device: str,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> None:
    """
    Train only the teacher's VQA head (linear probe) from cached h5 features.
    Florence-2 backbone never runs — features are read directly from the h5 cache.
    Trains and evaluates on the same 5k samples (no held-out split in mini-val setup).
    """
    import h5py

    cache_file = cfg["teacher"]["cache_file"]
    if not Path(cache_file).exists():
        print("  WARNING: teacher cache not found; skipping head training (accuracy will be 0%)")
        return

    samples_file = str(Path(cfg["data"]["cache_dir"]) / "vqa_samples.json")
    vocab_file = str(Path(cfg["data"]["cache_dir"]) / "answer_vocab.json")
    with open(samples_file) as f:
        samples = json.load(f)
    with open(vocab_file) as f:
        vocab = json.load(f)
    unk_id = vocab.get("<unk>", len(vocab) - 1)

    # Load cached last-layer features (index 3 = 4th hook = layer 5 in hook_layers=[0,2,3,5])
    last_feat_key = str(len(cfg["teacher"]["hook_layers"]) - 1)
    h5f = h5py.File(cache_file, "r")
    features, labels = [], []
    for s in samples:
        qid = str(int(s["question_id"]))
        if qid not in h5f:
            continue
        feat = torch.from_numpy(h5f[qid]["feats"][last_feat_key][:])  # (T, D)
        features.append(feat[0])  # BOS position → (D,)
        votes = Counter(a["answer"].lower().strip() for a in s["answers"])
        best_ans = votes.most_common(1)[0][0]
        labels.append(vocab.get(best_ans, unk_id))
    h5f.close()

    features = torch.stack(features).to(device)  # (N, D)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    N = features.size(0)

    print(f"  Training teacher VQA head on {N} cached samples for {epochs} epochs ...")
    teacher.vqa_head.train()
    optimizer = torch.optim.AdamW(teacher.vqa_head.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            logits = teacher.vqa_head(features[idx])
            loss = nn.functional.cross_entropy(logits, labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = teacher.vqa_head(features).argmax(-1)
                train_acc = (preds == labels).float().mean().item()
            print(f"    epoch {epoch+1}/{epochs}: loss={total_loss:.3f}  train_acc={train_acc:.4f}")

    teacher.vqa_head.eval()
    teacher.eval()


def vqa_accuracy(pred_ids: list, gt_ids: list) -> float:
    """Exact-match accuracy after answer-id comparison."""
    assert len(pred_ids) == len(gt_ids)
    return sum(p == g for p, g in zip(pred_ids, gt_ids)) / len(gt_ids)


@torch.no_grad()
def evaluate(model, loader, device: str, is_teacher: bool = False) -> float:
    model.eval()
    pred_ids, gt_ids = [], []

    for batch in loader:
        pv = batch["pixel_values"].to(device)
        ii = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        answers = batch["answer_id"].tolist()

        if is_teacher:
            logits, _, _ = model(pv, ii, am)
        else:
            logits, _, _ = model(pv, ii, am)

        preds = logits.argmax(dim=-1).cpu().tolist()
        pred_ids.extend(preds)
        gt_ids.extend(answers)

    acc = vqa_accuracy(pred_ids, gt_ids)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distil_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to student checkpoint")
    parser.add_argument("--teacher", action="store_true", help="Evaluate teacher instead")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_file = Path(cfg["data"]["cache_dir"]) / "answer_vocab.json"
    with open(vocab_file) as f:
        vocab = json.load(f)
    num_answers = len(vocab)

    loader = build_dataloader(
        samples_file=str(Path(cfg["data"]["cache_dir"]) / "vqa_samples.json"),
        vocab_file=str(vocab_file),
        image_dir=str(Path(cfg["data"]["data_dir"]) / "val2014"),
        processor_name=cfg["teacher"]["model_name"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=False,
    )

    if args.teacher:
        print("Evaluating teacher (Florence-2-base) ...")
        model = load_teacher(cfg, num_answers, device)
        print("Training teacher VQA head (linear probe on cached features) ...")
        train_teacher_head_from_cache(model, cfg, device)
        acc = evaluate(model, loader, device, is_teacher=True)
        print(f"Teacher VQA accuracy: {acc:.4f} ({acc*100:.2f}%)")
    else:
        if args.checkpoint is None:
            args.checkpoint = "checkpoints/best.pt"
        print(f"Evaluating student from {args.checkpoint} ...")
        student = build_student(cfg, num_answers)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        student.load_state_dict(ckpt["model_state_dict"])
        student = student.to(device)
        acc = evaluate(student, loader, device)
        print(f"Student VQA accuracy: {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()
