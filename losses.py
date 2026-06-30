"""
losses.py
=========
Multi-task loss for the proposed model and ablation variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PearsonCorrelation


class CorrelationWeightedLoss(nn.Module):
    """
    Multi-task loss combining:
      - sequence reconstruction loss
      - correlation-weighted scalar loss
      - error-correlation alignment loss
      - correlation distribution constraint loss

    Switches allow ablation studies to disable individual terms.
    """

    def __init__(
        self,
        seq_weight: float = 0.3,
        scalar_weight: float = 0.8,
        lambda_align: float = 0.1,
        low_corr_threshold: float = 0.3,
        high_corr_threshold: float = 0.7,
        realistic_max_corr: float = 0.95,
        soft_target: float = 0.92,
        lambda_linear: float = 0.1,
        lambda_soft: float = 0.02,
        lambda_exceed: float = 0.5,
        lambda_zero: float = 0.1,
        ema_decay: float = 0.99,
        use_corr_weighting: bool = True,
        use_align: bool = True,
        use_corr_dist: bool = True,
    ):
        super().__init__()
        self.seq_weight = seq_weight
        self.scalar_weight = scalar_weight
        self.lambda_align = lambda_align
        self.low_corr_threshold = low_corr_threshold
        self.high_corr_threshold = high_corr_threshold
        self.realistic_max_corr = realistic_max_corr
        self.soft_target = soft_target
        self.lambda_linear = lambda_linear
        self.lambda_soft = lambda_soft
        self.lambda_exceed = lambda_exceed
        self.lambda_zero = lambda_zero
        self.ema_decay = ema_decay

        self.use_corr_weighting = use_corr_weighting
        self.use_align = use_align
        self.use_corr_dist = use_corr_dist

        self.pearson = PearsonCorrelation()
        self.mse = nn.MSELoss(reduction="none")
        self.register_buffer("range_err_ema", torch.tensor(float("inf")))

    def forward(
        self,
        outputs: tuple,
        targets: tuple,
    ) -> tuple:
        seq1_pred, seq2_pred, mu = outputs
        seq1_true, seq2_true, scalar_true = targets
        device = mu.device

        has_seq = (seq1_pred is not None) and (seq2_pred is not None)

        # ---------- Sequence reconstruction loss ----------
        if has_seq:
            loss_seq1 = self.mse(seq1_pred, seq1_true).mean(dim=1).mean()
            loss_seq2 = self.mse(seq2_pred, seq2_true).mean(dim=1).mean()
            loss_seq = loss_seq1 * self.seq_weight + loss_seq2 * 0.05

            corr_feat = self.pearson(seq1_pred, seq2_pred)
            rho = corr_feat[:, 0]
            rho_abs = corr_feat[:, 1]
        else:
            loss_seq = torch.tensor(0.0, device=device)
            corr_feat = None
            rho = torch.zeros(mu.size(0), device=device)
            rho_abs = torch.zeros(mu.size(0), device=device)

        # ---------- Scalar loss ----------
        mse_per_sample = self.mse(mu, scalar_true).sum(dim=1)
        if self.use_corr_weighting and has_seq:
            sigmoid_weights = 0.3 + 0.4 * torch.sigmoid((rho_abs - 0.5) * 6)
            weights = torch.clamp(sigmoid_weights, max=0.75)
            weights = weights / (weights.mean() + 1e-8)
            weights = torch.clamp(weights, min=0.1)
            loss_scalar = (mse_per_sample * weights).mean() * self.scalar_weight
        else:
            loss_scalar = mse_per_sample.mean() * self.scalar_weight

        # ---------- Correlation distribution constraint ----------
        if self.use_corr_dist and has_seq:
            exceed_penalty = (
                F.relu(rho_abs - self.realistic_max_corr).mean() * self.lambda_exceed
            )
            mask_high_quality = (
                (rho_abs >= 0.9) & (rho_abs <= self.realistic_max_corr)
            ).float()
            soft_target_tensor = torch.full_like(rho_abs, self.soft_target)
            soft_penalty = (
                self.mse(rho_abs, soft_target_tensor) * mask_high_quality
            ).mean() * self.lambda_soft
            linear_penalty = F.relu(rho_abs - 0.9).mean() * self.lambda_linear
            zero_penalty = F.relu(0.3 - rho_abs).mean() * self.lambda_zero
            corr_dist_loss = (
                exceed_penalty + soft_penalty + linear_penalty + zero_penalty
            )
        else:
            corr_dist_loss = torch.tensor(0.0, device=device)

        # ---------- Alignment loss ----------
        if self.use_align and has_seq:
            with torch.no_grad():
                if self.training:
                    curr_range_mse = self.mse(
                        mu[:, 0:1], scalar_true[:, 0:1]
                    ).mean()
                    if curr_range_mse.isfinite():
                        if self.range_err_ema.device != curr_range_mse.device:
                            self.range_err_ema = self.range_err_ema.to(curr_range_mse.device)
                        if self.range_err_ema.isinf():
                            self.range_err_ema.copy_(curr_range_mse)
                        else:
                            self.range_err_ema.mul_(self.ema_decay).add_(
                                curr_range_mse, alpha=1 - self.ema_decay
                            )
                err_mean = mse_per_sample.mean() + 1e-8

            err_norm = mse_per_sample / err_mean
            badness = F.relu(err_norm - 1.0)
            batch_size = mse_per_sample.size(0)
            k = max(1, int(0.2 * batch_size))
            _, topk_idx = torch.topk(mse_per_sample, k, largest=False)
            rank_mask = torch.zeros_like(mse_per_sample)
            rank_mask[topk_idx] = 1.0

            range_rmse = self.mse(mu[:, 0:1], scalar_true[:, 0:1]).sum(dim=1).sqrt()
            if self.range_err_ema.isinf():
                sorted_rmse, _ = torch.sort(range_rmse)
                fallback_idx = max(1, int(0.1 * batch_size)) - 1
                fallback_thr = sorted_rmse[min(fallback_idx, batch_size - 1)]
                effective_thr = fallback_thr
            else:
                dynamic_thr = self.range_err_ema.sqrt() * 0.5
                sorted_rmse, _ = torch.sort(range_rmse)
                fallback_idx = max(1, int(0.1 * batch_size)) - 1
                fallback_thr = sorted_rmse[min(fallback_idx, batch_size - 1)]
                effective_thr = torch.min(dynamic_thr, fallback_thr)
            abs_mask = (range_rmse < effective_thr).float()

            seq_diff = ((seq1_pred - seq2_pred) ** 2).mean(dim=1)
            collusion = (rho_abs > 0.85) & (seq_diff < 1e-4)
            diversity_mask = (~collusion).float()
            good_mask = rank_mask * abs_mask * diversity_mask

            penalty = badness * rho_abs
            reward = good_mask * rho_abs
            alignment_loss = (penalty - 0.3 * reward).mean() * self.lambda_align
        else:
            alignment_loss = torch.tensor(0.0, device=device)

        # ---------- Total ----------
        total = loss_seq + loss_scalar
        if self.use_align and has_seq:
            total = total + alignment_loss
        if self.use_corr_dist and has_seq:
            total = total + corr_dist_loss

        with torch.no_grad():
            mean_rho = rho_abs.mean()
            extreme_ratio = (rho_abs > 0.9).float().mean()
            high_quality_ratio = (
                (rho_abs >= 0.9) & (rho_abs <= self.realistic_max_corr)
            ).float().mean()
            exceed_ratio = (rho_abs > self.realistic_max_corr).float().mean()

        metrics = {
            "seq": loss_seq.detach(),
            "scalar": loss_scalar.detach(),
            "corr_dist": corr_dist_loss.detach(),
            "alignment": alignment_loss.detach(),
            "mean_rho": mean_rho,
            "mean_raw_rho": rho.mean(),
            "weak_corr_ratio": (rho_abs < self.low_corr_threshold).float().mean(),
            "strong_corr_ratio": (
                (rho_abs >= self.high_corr_threshold) & (rho_abs <= 0.9)
            ).float().mean(),
            "extreme_corr_ratio": extreme_ratio,
            "exceed_ratio": exceed_ratio,
            "high_quality_ratio": high_quality_ratio,
        }
        if self.use_align and has_seq:
            metrics["reward_ratio"] = good_mask.mean()
            metrics["range_rmse_mean"] = range_rmse.mean()
        else:
            metrics["reward_ratio"] = torch.tensor(0.0, device=device)
            metrics["range_rmse_mean"] = torch.tensor(0.0, device=device)

        return total, metrics
