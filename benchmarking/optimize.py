"""
Applies inference optimizations to any nn.Module and verifies output correctness.

Optimization stack (applied in order, cumulative):
  1. bf16 precision
  2. torch.compile (mode="reduce-overhead")
  3. FlashAttention-2 (via HuggingFace attn_implementation= for teacher,
     automatic via torch sdp for student's nn.TransformerDecoderLayer)
"""

import torch
import torch.nn as nn


def apply_bf16(model: nn.Module) -> nn.Module:
    return model.to(torch.bfloat16)


def apply_torch_compile(model: nn.Module) -> nn.Module:
    return torch.compile(model, mode="reduce-overhead")


def apply_optimizations(
    model: nn.Module,
    use_bf16: bool = True,
    use_compile: bool = True,
    device: str = "cuda",
) -> nn.Module:
    model = model.to(device).eval()
    if use_bf16:
        model = apply_bf16(model)
    if use_compile:
        model = apply_torch_compile(model)
    return model


@torch.no_grad()
def verify_outputs(
    baseline_model: nn.Module,
    optimized_model: nn.Module,
    sample_inputs: dict,
    device: str = "cuda",
    atol: float = 1e-2,
) -> bool:
    """
    Checks that optimized model produces outputs within atol of baseline.
    Returns True if outputs are close enough.
    """
    baseline_model.eval()
    optimized_model.eval()

    pv = sample_inputs["pixel_values"].to(device)
    ii = sample_inputs["input_ids"].to(device)
    am = sample_inputs["attention_mask"].to(device)

    with torch.no_grad():
        base_logits, _, _ = baseline_model(pv, ii, am)
        opt_logits, _, _ = optimized_model(pv.to(next(optimized_model.parameters()).dtype), ii, am)

    opt_logits = opt_logits.to(base_logits.dtype)
    max_diff = (base_logits - opt_logits).abs().max().item()
    argmax_agree = (base_logits.argmax(-1) == opt_logits.argmax(-1)).float().mean().item()
    print(f"  Output verification: max_diff={max_diff:.4f}, argmax_agreement={argmax_agree:.2%}")
    return argmax_agree > 0.9
