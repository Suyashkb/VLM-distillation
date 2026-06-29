"""
Distillation loss components:
  L_task  = CrossEntropy(student_logits, gt_answer_id)
  L_feat  = mean MSE between projected student features and teacher features
  L_attn  = mean KL-divergence between student and teacher attention maps
  L_total = alpha * L_task + beta * L_feat + gamma * L_attn
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def task_loss(
        self, student_logits: torch.Tensor, answer_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.ce(student_logits, answer_ids)

    def feature_loss(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        if not student_feats or not teacher_feats:
            return torch.tensor(0.0, device=student_feats[0].device if student_feats else "cpu")

        losses = []
        n_pairs = min(len(student_feats), len(teacher_feats))
        for sf, tf in zip(student_feats[:n_pairs], teacher_feats[:n_pairs]):
            # Align sequence lengths via adaptive pooling if they differ
            if sf.shape[1] != tf.shape[1]:
                sf = sf.transpose(1, 2)  # (B, D, T)
                sf = F.adaptive_avg_pool1d(sf, tf.shape[1])
                sf = sf.transpose(1, 2)
            losses.append(F.mse_loss(sf, tf.detach()))
        return torch.stack(losses).mean()

    def attention_loss(
        self,
        student_attns: List[torch.Tensor],
        teacher_attns: List[torch.Tensor],
    ) -> torch.Tensor:
        if not student_attns or not teacher_attns:
            return torch.tensor(0.0)

        losses = []
        n_pairs = min(len(student_attns), len(teacher_attns))
        for sa, ta in zip(student_attns[:n_pairs], teacher_attns[:n_pairs]):
            # sa, ta: (B, heads, T, T) — average over heads
            sa = sa.mean(1)  # (B, T, T)
            ta = ta.mean(1).detach()

            # Align sequence lengths if needed
            if sa.shape != ta.shape:
                T_t = ta.shape[-1]
                sa = F.adaptive_avg_pool2d(sa.unsqueeze(1), (T_t, T_t)).squeeze(1)

            sa_log = F.log_softmax(sa.flatten(1), dim=-1)
            ta_prob = F.softmax(ta.flatten(1), dim=-1)
            losses.append(F.kl_div(sa_log, ta_prob, reduction="batchmean"))
        return torch.stack(losses).mean()

    def forward(
        self,
        student_logits: torch.Tensor,
        answer_ids: torch.Tensor,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
        student_attns: Optional[List[torch.Tensor]] = None,
        teacher_attns: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        l_task = self.task_loss(student_logits, answer_ids)
        l_feat = self.feature_loss(student_feats, teacher_feats)
        l_attn = self.attention_loss(
            student_attns or [], teacher_attns or []
        )
        total = self.alpha * l_task + self.beta * l_feat + self.gamma * l_attn
        return total, {
            "loss": total.item(),
            "l_task": l_task.item(),
            "l_feat": l_feat.item(),
            "l_attn": l_attn.item(),
        }
