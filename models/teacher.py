"""
Frozen Florence-2-base teacher with forward hooks for distillation.

Exposes intermediate features and attention maps from specified decoder layers.
Caches outputs to HDF5 so training never re-runs the 230M frozen model.
"""

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


class TeacherVLM(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        hook_layers: list = None,
        num_answers: int = 3130,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.hook_layers = hook_layers or [3, 5, 7, 9]

        _rev = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, revision=_rev)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=_rev,
        ).to(device)

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        # VQA classification head (trained alongside student for teacher accuracy measurement)
        hidden_size = self.model.config.text_config.d_model
        self.vqa_head = nn.Linear(hidden_size, num_answers).to(device)

        self._feats: list = []
        self._attns: list = []
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        decoder_layers = self.model.language_model.model.decoder.layers
        for layer_idx in self.hook_layers:
            layer = decoder_layers[layer_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # output is (hidden_states, self_attn_weights, ...)
                    hidden = output[0]   # (B, T, D)
                    # attn weights only available when output_attentions=True
                    attn = output[1] if len(output) > 1 and output[1] is not None else None
                    self._feats.append(hidden.detach())
                    if attn is not None:
                        self._attns.append(attn.detach())
                return hook

            h = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self._feats.clear()
        self._attns.clear()

        # Florence-2 is seq2seq: input_ids → encoder, decoder needs decoder_input_ids.
        # We pass a single BOS token so the decoder runs one step and our hooks fire.
        bos_id = self.model.config.decoder_start_token_id or self.model.config.bos_token_id
        decoder_input_ids = torch.full(
            (pixel_values.size(0), 1), bos_id, dtype=torch.long, device=pixel_values.device
        )

        with torch.no_grad():
            self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
                return_dict=True,
            )
        # Use last hooked decoder layer, position 0 (the BOS decoder step) for VQA head.
        if self._feats:
            last_hidden = self._feats[-1][:, 0, :]
        else:
            raise RuntimeError("No features captured — check hook_layers indices against model depth.")
        logits = self.vqa_head(last_hidden)
        return logits, list(self._feats), list(self._attns)

    @property
    def hidden_size(self) -> int:
        return self.model.config.text_config.d_model


@torch.no_grad()
def cache_teacher_features(
    teacher: TeacherVLM,
    dataloader,
    cache_file: str,
    device: str = "cuda",
) -> None:
    """Run teacher once over entire dataset and save features to HDF5."""
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Teacher cache already exists: {cache_file}")
        return

    # Write to a temp file and rename on completion so a crash never leaves a stale skeleton.
    tmp_path = cache_path.with_suffix(".h5.tmp")
    tmp_path.unlink(missing_ok=True)

    teacher.eval()
    try:
        with h5py.File(tmp_path, "w") as h5f:
            for batch in tqdm(dataloader, desc="Caching teacher features"):
                pv = batch["pixel_values"].to(device)
                ii = batch["input_ids"].to(device)
                am = batch["attention_mask"].to(device)
                qids = batch["question_id"]

                _, feats, attns = teacher(pv, ii, am)

                for b_idx, qid in enumerate(qids):
                    key = str(qid.item() if hasattr(qid, "item") else qid)
                    grp = h5f.require_group(key)
                    feat_grp = grp.require_group("feats")
                    attn_grp = grp.require_group("attns")
                    for i, f in enumerate(feats):
                        feat_grp.create_dataset(str(i), data=f[b_idx].cpu().numpy())
                    for i, a in enumerate(attns):
                        attn_grp.create_dataset(str(i), data=a[b_idx].cpu().numpy())

        tmp_path.rename(cache_path)
        print(f"Saved teacher cache → {cache_file}")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_teacher(cfg: dict, num_answers: int, device: str = "cuda") -> TeacherVLM:
    return TeacherVLM(
        model_name=cfg["teacher"]["model_name"],
        hook_layers=cfg["teacher"]["hook_layers"],
        num_answers=num_answers,
        device=device,
    )
