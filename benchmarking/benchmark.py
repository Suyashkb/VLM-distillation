"""
Throughput + latency benchmarking for teacher and student under various optimization configs.

Outputs:
  results/benchmark_results.json  — raw numbers
  results/benchmark_summary.csv   — human-readable table

Usage:
    python benchmarking/benchmark.py --config configs/distil_config.yaml \
        --student_checkpoint checkpoints/best.pt
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
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.optimize import apply_optimizations, verify_outputs
from data.vqa_dataset import build_dataloader
from models.student import build_student
from models.teacher import load_teacher


def make_dummy_batch(pixel_values, input_ids, attention_mask, batch_size: int, device: str):
    return (
        pixel_values[:batch_size].to(device),
        input_ids[:batch_size].to(device),
        attention_mask[:batch_size].to(device),
    )


@torch.no_grad()
def measure(
    model,
    pv: torch.Tensor,
    ii: torch.Tensor,
    am: torch.Tensor,
    warmup: int = 10,
    repeats: int = 100,
) -> dict:
    dtype = next(model.parameters()).dtype
    pv = pv.to(dtype)

    # Warmup (ensures torch.compile graph is built)
    for _ in range(warmup):
        _ = model(pv, ii, am)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model(pv, ii, am)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    import statistics
    latency_ms = [t * 1000 for t in times]
    mean_ms = statistics.mean(latency_ms)
    std_ms = statistics.stdev(latency_ms)
    throughput = pv.size(0) / (mean_ms / 1000)  # samples/sec

    return {
        "latency_mean_ms": round(mean_ms, 3),
        "latency_std_ms": round(std_ms, 3),
        "throughput_samples_per_sec": round(throughput, 1),
    }


def run_benchmark(cfg: dict, student_checkpoint: str) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
    warmup = cfg["benchmarking"]["warmup_batches"]
    repeats = cfg["benchmarking"]["timing_batches"]
    batch_sizes = cfg["benchmarking"]["batch_sizes"]

    vocab_file = Path(cfg["data"]["cache_dir"]) / "answer_vocab.json"
    with open(vocab_file) as f:
        vocab = json.load(f)
    num_answers = len(vocab)

    # Load one batch of real data for inputs
    loader = build_dataloader(
        samples_file=str(Path(cfg["data"]["cache_dir"]) / "vqa_samples.json"),
        vocab_file=str(vocab_file),
        image_dir=str(Path(cfg["data"]["data_dir"]) / "val2014"),
        processor_name=cfg["teacher"]["model_name"],
        batch_size=max(batch_sizes),
        num_workers=0,
        shuffle=False,
    )
    ref_batch = next(iter(loader))
    pv_ref = ref_batch["pixel_values"]
    ii_ref = ref_batch["input_ids"]
    am_ref = ref_batch["attention_mask"]

    results = []

    configs = [
        {"use_bf16": False, "use_compile": False, "label": "FP32 baseline"},
        {"use_bf16": True,  "use_compile": False, "label": "bf16"},
        {"use_bf16": True,  "use_compile": True,  "label": "bf16 + compile"},
    ]

    for model_name, loader_fn in [
        ("teacher", lambda: _load_teacher_seeded(cfg, num_answers, device)),
        ("student", lambda: _load_student(cfg, num_answers, student_checkpoint, device)),
    ]:
        print(f"\n{'='*50}\nBenchmarking: {model_name}\n{'='*50}")

        for opt_cfg in configs:
            label = opt_cfg["label"]
            print(f"\n  Config: {label}")

            model = loader_fn()
            baseline_model = loader_fn() if opt_cfg["use_bf16"] or opt_cfg["use_compile"] else None

            model = apply_optimizations(
                model,
                use_bf16=opt_cfg["use_bf16"],
                use_compile=opt_cfg["use_compile"],
                device=device,
            )

            # Verify outputs before timing
            if baseline_model is not None:
                baseline_model = apply_optimizations(baseline_model, use_bf16=False, use_compile=False, device=device)
                sample = {"pixel_values": pv_ref[:2], "input_ids": ii_ref[:2], "attention_mask": am_ref[:2]}
                verify_outputs(baseline_model, model, sample, device)
                del baseline_model

            for bs in batch_sizes:
                if bs > pv_ref.size(0):
                    continue
                pv, ii, am = make_dummy_batch(pv_ref, ii_ref, am_ref, bs, device)
                try:
                    metrics = measure(model, pv, ii, am, warmup=warmup, repeats=repeats)
                except Exception as e:
                    metrics = {"error": str(e)}

                row = {
                    "model": model_name,
                    "optimization": label,
                    "batch_size": bs,
                    **metrics,
                }
                results.append(row)
                if "error" not in metrics:
                    print(
                        f"  bs={bs}: latency={metrics['latency_mean_ms']:.1f}±{metrics['latency_std_ms']:.1f}ms "
                        f"throughput={metrics['throughput_samples_per_sec']:.0f} samples/s"
                    )
                else:
                    print(f"  bs={bs}: ERROR — {metrics['error']}")

            del model
            torch.cuda.empty_cache()

    return results


def _load_teacher_seeded(cfg, num_answers, device):
    """Load teacher with a fixed seed so the VQA head is identical across calls (for verify_outputs)."""
    torch.manual_seed(0)
    teacher = load_teacher(cfg, num_answers, device)
    torch.seed()  # restore random state
    return teacher


def _load_student(cfg, num_answers, checkpoint_path, device):
    student = build_student(cfg, num_answers)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    student.load_state_dict(ckpt["model_state_dict"])
    return student.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distil_config.yaml")
    parser.add_argument("--student_checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_benchmark(cfg, args.student_checkpoint)

    results_dir = Path(cfg["benchmarking"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = results_dir / "benchmark_summary.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved → {json_path}")
    print(f"Summary table → {csv_path}")


if __name__ == "__main__":
    main()
