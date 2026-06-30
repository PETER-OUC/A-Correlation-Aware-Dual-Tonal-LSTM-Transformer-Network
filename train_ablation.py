"""
train_ablation.py
=================
统一消融实验训练脚本。

功能：
- 读取基础 YAML/JSON 配置文件（如 configs/ablation_run.yaml）。
- 按配置训练指定的消融变体，每个变体保存在独立文件夹。
- 自动在路径名中加入日期时间，避免覆盖历史结果。
- 保存 best_model.pth，并每 10 个 epoch 保存一次 epoch_xxxx.pth。
- 每行日志都打印当前 epoch、训练/验证损失、学习率等，方便看进度。

使用方法：

    # 训练全部消融变体
    python train_ablation.py --config configs/ablation_run.yaml --modes all

    # 训练指定变体（例如审稿人要求的 5 个）
    python train_ablation.py --config configs/ablation_run.yaml ^
        --modes full no_gated_fusion no_physics lstm_only transformer_only

    # 自定义保存根目录（不指定则自动生成带时间戳的目录）
    python train_ablation.py --config configs/ablation_run.yaml ^
        --modes full --save_root checkpoints/my_ablation

保存路径示例：
    checkpoints/ablation_batch_20250625_143052/ablation_full_20250625_143052/
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from losses import CorrelationWeightedLoss
from model import build_model


ABLATION_MODES = {
    "full": {
        "description": "Full proposed model",
        "use_input2": True,
        "use_transformer": True,
        "use_physics": True,
        "use_gated_fusion": True,
        "use_corr_embed": True,
        "use_corr_weighting": True,
        "use_align": True,
        "use_corr_dist": True,
    },
    "no_physics": {
        "description": "No physics-informed correlation layer",
        "use_input2": True,
        "use_transformer": True,
        "use_physics": False,
        "use_gated_fusion": False,
        "use_corr_embed": False,
        "use_corr_weighting": False,
        "use_align": False,
        "use_corr_dist": False,
    },
    "no_gated_fusion": {
        "description": "Replace gated fusion with simple addition",
        "use_input2": True,
        "use_transformer": True,
        "use_physics": True,
        "use_gated_fusion": False,
        "use_corr_embed": True,
        "use_corr_weighting": True,
        "use_align": True,
        "use_corr_dist": True,
    },
    "no_dual_branch": {
        "description": "Remove auxiliary input branch",
        "use_input2": False,
        "use_transformer": True,
        "use_physics": True,
        "use_gated_fusion": True,
        "use_corr_embed": True,
        "use_corr_weighting": True,
        "use_align": True,
        "use_corr_dist": True,
    },
    "no_correlation_loss": {
        "description": "Keep physics layer but remove correlation distribution loss",
        "use_input2": True,
        "use_transformer": True,
        "use_physics": True,
        "use_gated_fusion": True,
        "use_corr_embed": True,
        "use_corr_weighting": True,
        "use_align": True,
        "use_corr_dist": False,
    },
    "simple_concat": {
        "description": "Replace gated fusion with simple concatenation/addition",
        "use_input2": True,
        "use_transformer": True,
        "use_physics": True,
        "use_gated_fusion": False,
        "use_corr_embed": True,
        "use_corr_weighting": True,
        "use_align": True,
        "use_corr_dist": True,
    },
    "lstm_only": {
        "description": "LSTM-only backbone with physics-informed correlation layer",
        "use_input2": True,
        "use_transformer": False,
        "use_physics": True,
        "use_gated_fusion": True,
        "use_corr_embed": True,
        "use_corr_weighting": True,
        "use_align": True,
        "use_corr_dist": True,
    },
    "transformer_only": {
        "description": "Transformer-only backbone with physics-informed correlation layer",
        "model_type": "transformer_only",
        "use_physics": True,
        "use_gated_fusion": True,
        "use_corr_embed": True,
        "use_corr_weighting": True,
        "use_align": True,
        "use_corr_dist": True,
    },
}


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_unified_config(config: dict, save_dir: str):
    """Save the merged config in YAML and JSON."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def save_split_configs(config: dict, save_dir: str):
    """Also save legacy split configs for compatibility with older plotting scripts."""
    os.makedirs(save_dir, exist_ok=True)

    model_keys = {
        "input_sizes", "seq_lens", "lstm_hiddens", "target_seq_len", "d_model",
        "nhead", "num_layers", "dim_feedforward", "lstm_layers", "bidirectional",
        "dropout", "num_scalars", "model_type", "use_input2", "use_transformer",
        "use_physics", "use_gated_fusion", "use_corr_embed",
    }
    data_keys = {
        "data_path", "env_range", "file_range", "sample_num_str", "seq_lens",
        "input_sizes", "target_seq_len", "range_min", "range_max", "vel_min",
        "vel_max", "single_spectrum",
    }
    train_keys = set(config.keys()) - model_keys - data_keys

    model_cfg = {k: config[k] for k in model_keys if k in config}
    data_cfg = {k: config[k] for k in data_keys if k in config}
    train_cfg = {k: config[k] for k in train_keys if k in config}

    for name, cfg in [
        ("model_config", model_cfg),
        ("data_config", data_cfg),
        ("train_config", train_cfg),
    ]:
        with open(os.path.join(save_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)


def save_configs(config: dict, save_dir: str):
    save_unified_config(config, save_dir)
    save_split_configs(config, save_dir)


def build_optimizer(model: torch.nn.Module, config: dict):
    return optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer: optim.Optimizer, config: dict):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.get("scheduler_patience", 20),
        min_lr=1e-7,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: float,
    mode: str,
    epoch: int,
    num_epochs: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    comp_keys = ["seq", "scalar", "corr_dist", "alignment", "mean_rho"]
    comp_sum = {k: 0.0 for k in comp_keys}
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"[{mode}] Epoch {epoch+1}/{num_epochs}",
        leave=False,
    )
    for X1, X2, seq1, seq2, scalar in pbar:
        X1 = X1.to(device, non_blocking=True)
        X2 = X2.to(device, non_blocking=True)
        seq1 = seq1.to(device, non_blocking=True)
        seq2 = seq2.to(device, non_blocking=True)
        scalar = scalar.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            seq1_pred, seq2_pred, _, _, mu, _ = model(X1, X2)
            loss, metrics = criterion(
                (seq1_pred, seq2_pred, mu), (seq1, seq2, scalar)
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        for k in comp_keys:
            comp_sum[k] += metrics[k].item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {k: v / n_batches for k, v in comp_sum.items()} | {
        "loss": total_loss / n_batches
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module, loader, criterion, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    comp_keys = ["seq", "scalar", "corr_dist", "alignment", "mean_rho"]
    comp_sum = {k: 0.0 for k in comp_keys}
    n_batches = 0

    for X1, X2, seq1, seq2, scalar in loader:
        X1 = X1.to(device, non_blocking=True)
        X2 = X2.to(device, non_blocking=True)
        seq1 = seq1.to(device, non_blocking=True)
        seq2 = seq2.to(device, non_blocking=True)
        scalar = scalar.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            seq1_pred, seq2_pred, _, _, mu, _ = model(X1, X2)
            loss, metrics = criterion(
                (seq1_pred, seq2_pred, mu), (seq1, seq2, scalar)
            )

        total_loss += loss.item()
        for k in comp_keys:
            comp_sum[k] += metrics[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in comp_sum.items()} | {
        "loss": total_loss / n_batches
    }


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    val_loss: float,
    patience_counter: int,
    config: dict,
    history: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "val_loss": val_loss,
            "patience_counter": patience_counter,
            "config": config,
            "history": history,
        },
        path,
    )


def run_ablation(
    base_config: dict,
    mode: str,
    save_root: str,
    device: torch.device,
    resume_dir: Optional[str] = None,
):
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode}")

    mode_cfg = ABLATION_MODES[mode]
    description = mode_cfg.get("description", "")

    if resume_dir:
        save_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(save_dir):
            raise ValueError(f"Resume directory does not exist: {save_dir}")
        print("\n" + "=" * 64)
        print(f"Resuming mode : {mode}")
        print(f"Save directory: {save_dir}")
        print("=" * 64)
    else:
        config = {**base_config, **mode_cfg}
        config["ablation_mode"] = mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_root, f"ablation_{mode}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        save_configs(config, save_dir)

        print("\n" + "=" * 64)
        print(f"Ablation mode : {mode}")
        print(f"Description   : {description}")
        print(f"Save directory: {save_dir}")
        print("=" * 64)

    # Resolve config and optional resume checkpoint
    if resume_dir:
        ckpt_path = os.path.join(save_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            periodic = sorted(
                [
                    f
                    for f in os.listdir(save_dir)
                    if f.startswith("epoch_") and f.endswith(".pth")
                ]
            )
            if periodic:
                ckpt_path = os.path.join(save_dir, periodic[-1])
            else:
                raise FileNotFoundError(f"No checkpoint found in {save_dir}")
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt.get("config")
        if config is None:
            config = {**base_config, **mode_cfg}
            config["ablation_mode"] = mode
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", {"train": [], "val": []})
        resume_ckpt = ckpt
    else:
        config = {**base_config, **mode_cfg}
        config["ablation_mode"] = mode
        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train": [], "val": []}
        resume_ckpt = None

    # Data (lazy import so the script can still show --help even if scikit-learn is slow to load)
    from dataset import build_dataloaders

    # Data
    train_loader, val_loader, _ = build_dataloaders(
        data_path=config["data_path"],
        seq_lens=config.get("seq_lens", [400, 1400]),
        input_sizes=config.get("input_sizes", [5, 2]),
        env_range=tuple(config.get("env_range", [1, 12])),
        file_range=tuple(config.get("file_range", [1, 6])),
        sample_num_str=config.get("sample_num_str", "0005e3"),
        test_size=config.get("test_size", 0.1),
        val_size=config.get("val_size", 0.1),
        batch_size=config.get("batch_size", 512),
        num_workers=config.get("num_workers", 0),
        single_spectrum=config.get("single_spectrum", False),
        seed=config.get("seed", 42),
    )
    print(
        f"Data loaders  : train={len(train_loader)} | val={len(val_loader)} batches"
    )

    # Model
    model = build_model(config).to(device)
    print(f"Model params  : {model.count_parameters():,}")

    # Loss
    criterion = CorrelationWeightedLoss(
        seq_weight=config.get("seq_weight", 0.3),
        scalar_weight=config.get("scalar_weight", 0.8),
        lambda_align=config.get("lambda_align", 0.1),
        low_corr_threshold=config.get("low_corr_threshold", 0.3),
        high_corr_threshold=config.get("high_corr_threshold", 0.7),
        realistic_max_corr=config.get("realistic_max_corr", 0.95),
        soft_target=config.get("soft_target", 0.92),
        lambda_linear=config.get("lambda_linear", 0.1),
        lambda_soft=config.get("lambda_soft", 0.02),
        lambda_exceed=config.get("lambda_exceed", 0.5),
        lambda_zero=config.get("lambda_zero", 0.1),
        ema_decay=config.get("ema_decay", 0.99),
        use_corr_weighting=config.get("use_corr_weighting", True),
        use_align=config.get("use_align", True),
        use_corr_dist=config.get("use_corr_dist", True),
    ).to(device)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler() if device.type == "cuda" else None

    # Restore states if resuming
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if scaler is not None and resume_ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        print(
            f"Resumed from epoch {start_epoch} | best_val={best_val_loss:.6f} | "
            f"patience_counter={patience_counter}"
        )

    num_epochs = config.get("num_epochs", 50)
    grad_clip = config.get("grad_clip", 0.5)
    patience = config.get("patience", 200)
    save_every = config.get("save_every", 10)

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, grad_clip,
            mode, epoch + 1, num_epochs,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        best_flag = ""
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(
                os.path.join(save_dir, "best_model.pth"),
                model, optimizer, scheduler, scaler, epoch, best_val_loss,
                patience_counter, config, history,
            )
            best_flag = " | *best"
        else:
            patience_counter += 1

        # Lightweight periodic checkpoint (every N epochs)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                os.path.join(save_dir, f"epoch_{epoch+1:04d}.pth"),
                model, optimizer, scheduler, scaler, epoch, val_metrics["loss"],
                patience_counter, config, history,
            )

        # One-line epoch summary
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"train={train_metrics['loss']:.6f} | "
            f"val={val_metrics['loss']:.6f} | "
            f"lr={current_lr:.2e} | "
            f"seq={train_metrics['seq']:.4f} | "
            f"scalar={train_metrics['scalar']:.4f} | "
            f"align={train_metrics['alignment']:.4f} | "
            f"corr_dist={train_metrics['corr_dist']:.4f} | "
            f"|rho|={train_metrics['mean_rho']:.4f}"
            f"{best_flag}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    print(f"\nFinished {mode}: best_val={best_val_loss:.6f} | time={elapsed/60:.1f}min")
    print(f"Checkpoints saved under: {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Train ablation variants")
    parser.add_argument("--config", default="configs/config.yaml", help="Base config YAML/JSON")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(ABLATION_MODES.keys()) + ["all"],
        default=["all"],
        help="Ablation modes to train (space-separated, or 'all')",
    )
    parser.add_argument(
        "--save_root",
        default=None,
        help="Root directory where ablation_* folders are created (default: auto timestamp)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing ablation_*/ folder to resume training from",
    )
    args = parser.parse_args()

    # Load base config
    with open(args.config, "r", encoding="utf-8") as f:
        if args.config.endswith(".json"):
            base_config = json.load(f)
        else:
            base_config = yaml.safe_load(f)

    set_seed(base_config.get("seed", 42))

    resume_dir = os.path.abspath(args.resume) if args.resume else None
    if resume_dir and not os.path.isdir(resume_dir):
        raise ValueError(f"Resume directory does not exist: {resume_dir}")

    if args.save_root is None and resume_dir is None:
        args.save_root = os.path.join(
            "checkpoints", f"ablation_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    if args.save_root is not None:
        os.makedirs(args.save_root, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    modes = list(ABLATION_MODES.keys()) if "all" in args.modes else args.modes
    if resume_dir and len(modes) > 1:
        raise ValueError("--resume can only be used with a single mode")
    print(f"\nWill train {len(modes)} mode(s): {', '.join(modes)}")
    if resume_dir:
        print(f"Resume dir: {resume_dir}\n")
    else:
        print(f"Save root: {args.save_root}\n")

    for mode in modes:
        run_ablation(base_config, mode, args.save_root, device, resume_dir=resume_dir)

    print("\nAll ablation training complete.")


if __name__ == "__main__":
    main()
