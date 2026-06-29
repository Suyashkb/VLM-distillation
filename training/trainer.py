"""
Distillation trainer: runs training + validation loops with checkpointing.
"""

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.distillation_losses import DistillationLoss


class Trainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: Optional[nn.Module],
        train_loader,
        val_loader,
        cfg: dict,
        device: str = "cuda",
    ):
        self.student = student.to(device)
        self.teacher = teacher
        if self.teacher is not None:
            self.teacher.eval()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        tc = cfg["training"]
        dc = cfg["distillation"]

        self.epochs = tc["epochs"]
        self.grad_clip = tc["grad_clip"]
        self.log_interval = tc["log_interval"]
        self.val_freq = tc["val_freq"]
        self.ckpt_dir = Path(tc["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = DistillationLoss(
            alpha=dc["alpha"], beta=dc["beta"], gamma=dc["gamma"]
        )

        self.optimizer = AdamW(
            self.student.parameters(),
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )

        total_steps = self.epochs * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
        self.warmup_steps = tc["warmup_steps"]

        self.history = {"train": [], "val": []}
        self.best_val_acc = 0.0
        self.global_step = 0

    def _warmup_lr(self) -> None:
        if self.global_step < self.warmup_steps:
            lr = self.cfg["training"]["learning_rate"] * (self.global_step / max(self.warmup_steps, 1))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _train_epoch(self, epoch: int) -> dict:
        self.student.train()
        running = {"loss": 0.0, "l_task": 0.0, "l_feat": 0.0, "l_attn": 0.0}
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [train]")
        for step, batch in enumerate(pbar):
            pv = batch["pixel_values"].to(self.device)
            ii = batch["input_ids"].to(self.device)
            am = batch["attention_mask"].to(self.device)
            answer_ids = batch["answer_id"].to(self.device)

            # Get teacher features: either from cache (in batch) or live inference
            if "teacher_feats" in batch:
                t_feats = [f.to(self.device) for f in batch["teacher_feats"]]
                t_attns = [a.to(self.device) for a in batch.get("teacher_attns", [])]
            elif self.teacher is not None:
                with torch.no_grad():
                    _, t_feats, t_attns = self.teacher(pv, ii, am)
            else:
                t_feats, t_attns = [], []

            # Student forward
            s_logits, s_feats, s_attns = self.student(pv, ii, am)

            loss, metrics = self.loss_fn(
                s_logits, answer_ids, s_feats, t_feats, s_attns, t_attns
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
            self.optimizer.step()
            self._warmup_lr()
            if self.global_step >= self.warmup_steps:
                self.scheduler.step()
            self.global_step += 1

            for k, v in metrics.items():
                running[k] += v
            pred = s_logits.argmax(dim=-1)
            correct += (pred == answer_ids).sum().item()
            total += answer_ids.size(0)

            if (step + 1) % self.log_interval == 0:
                avg_loss = running["loss"] / (step + 1)
                acc = correct / total
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.3f}")

        n = len(self.train_loader)
        return {k: v / n for k, v in running.items()} | {"acc": correct / total}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        self.student.eval()
        running = {"loss": 0.0, "l_task": 0.0, "l_feat": 0.0, "l_attn": 0.0}
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [val]"):
            pv = batch["pixel_values"].to(self.device)
            ii = batch["input_ids"].to(self.device)
            am = batch["attention_mask"].to(self.device)
            answer_ids = batch["answer_id"].to(self.device)

            if "teacher_feats" in batch:
                t_feats = [f.to(self.device) for f in batch["teacher_feats"]]
                t_attns = [a.to(self.device) for a in batch.get("teacher_attns", [])]
            elif self.teacher is not None:
                _, t_feats, t_attns = self.teacher(pv, ii, am)
            else:
                t_feats, t_attns = [], []

            s_logits, s_feats, s_attns = self.student(pv, ii, am)
            _, metrics = self.loss_fn(s_logits, answer_ids, s_feats, t_feats, s_attns, t_attns)

            for k, v in metrics.items():
                running[k] += v
            pred = s_logits.argmax(dim=-1)
            correct += (pred == answer_ids).sum().item()
            total += answer_ids.size(0)

        n = len(self.val_loader)
        return {k: v / n for k, v in running.items()} | {"acc": correct / total}

    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        torch.save(state, self.ckpt_dir / "last.pt")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")
            print(f"  ** New best val acc: {val_metrics['acc']:.4f}")

    def fit(self) -> None:
        for epoch in range(self.epochs):
            train_metrics = self._train_epoch(epoch)
            self.history["train"].append(train_metrics)

            print(
                f"Epoch {epoch+1} | train loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} | "
                f"l_task={train_metrics['l_task']:.4f} "
                f"l_feat={train_metrics['l_feat']:.4f}"
            )

            if (epoch + 1) % self.val_freq == 0:
                val_metrics = self._val_epoch(epoch)
                self.history["val"].append({"epoch": epoch} | val_metrics)
                is_best = val_metrics["acc"] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics["acc"]
                self._save_checkpoint(epoch, val_metrics, is_best)
                print(
                    f"  val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}"
                )

        # Save training history for plotting
        with open(self.ckpt_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete. Best val acc: {self.best_val_acc:.4f}")
