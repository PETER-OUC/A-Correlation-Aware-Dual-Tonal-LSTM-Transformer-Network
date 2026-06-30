r"""
predict_ablation.py
===================
Batch prediction & visualization for ablation study checkpoints.

功能：
- 对 --checkpoint_root 下每个 ablation_* 文件夹中的 .pth 检查点进行批量评估。
- 默认评估 best_model.pth 以及所有 epoch_*.pth（可用 --best_only 只评估 best）。
- 为每个检查点生成 7 张单模型图（与 evaluate_SW96_two_V_R.py 对应）。
- 按“检查点类型（stem）”分组生成跨变体对比图，例如：
      comparison/best_model/          ← 仅包含各 mode 的 best_model
      comparison/epoch_0010/          ← 仅包含各 mode 的 epoch_0010
      comparison/epoch_0020/          ← ...
    - 对比图包含：误差对比、相关系数/参数量、Range/Velocity 散点、阈值误差、
      以及新增的“各相关系数阈值范围内样本数分布”图。
- 支持通过 --stems 指定只评估某些检查点类型，例如：
      --stems best_model
      --stems best_model epoch_0010 epoch_0020

输出结构示例：
    predictions/
    └── ablation_batch_20260628_144355/
        ├── SW96_all/
        │   ├── full_best_model/
        │   ├── full_epoch_0010/
        │   ├── no_gated_fusion_best_model/
        │   │   ...
        │   ├── comparison/
        │   │   ├── best_model/
        │   │   │   ├── SW96_all_best_model_comparison_01_error_comparison.png
        │   │   │   ├── ...
        │   │   │   └── SW96_all_best_model_results.xlsx
        │   │   ├── epoch_0010/
        │   │   │   └── ...
        │   │   └── epoch_0050/
        │   │       └── ...
        │   └── ...
        └── SW96_h01/
            └── ...

用法示例：
    # 评估所有检查点，并按 stem 分组输出对比图
    python predict_ablation.py  --checkpoint_root checkpoints/ablation_run_0629  --sw96_dir "E:\\Moving\\数据制作\\SW96_数据集" --output_dir "predictions/ablation_run_0629" --stems best_model

    # 只用 best_model 做预测和对比
    python predict_ablation.py \
        --checkpoint_root checkpoints/ablation_batch_20260628_144355 \
        --sw96_dir "E:\\Moving\\数据制作\\SW96_数据集" \
        --output_dir predictions/ablation_batch_20260628_144355 \
        --stems best_model

    # 只用 best_model 和 epoch_0010
    python predict_ablation.py \
        --checkpoint_root checkpoints/ablation_batch_20260628_144355 \
        --sw96_dir "E:\\Moving\\数据制作\\SW96_数据集" \
        --output_dir predictions/ablation_batch_20260628_144355 \
        --stems best_model epoch_0010
"""

import argparse
import json
import os
import re
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from evaluate import evaluate_model, load_config, load_model

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = True

BATCH_SIZE = 64
FIGURE_DPI = 150
WINDOW_SIZE = 50

# Display-friendly labels for ablation modes (used in plots/legends/tables).
# Keys must match the mode names produced by train_ablation.py.
MODE_LABELS = {
    "full": "Full model",
    "no_physics": "Without physics layer",
    "no_gated_fusion": "Without gated fusion",
    "no_dual_branch": "Without dual branch",
    "no_correlation_loss": "Without correlation loss",
    "simple_concat": "Simple concat fusion",
    "lstm_only": "LSTM only",
    "transformer_only": "Transformer only",
}


def mode_label(mode: str) -> str:
    """Return a publication-ready label for an ablation mode key."""
    return MODE_LABELS.get(mode, mode)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def discover_ablation_dirs(root: str) -> List[str]:
    """Return ablation_* directories under root, or root itself if it is one."""
    root = os.path.abspath(root)
    dirs = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and d.startswith("ablation_"):
            dirs.append(p)
    if dirs:
        return dirs
    if os.path.basename(root).startswith("ablation_") and os.path.isdir(root):
        return [root]
    return []


def discover_sw96_files(sw96_dir: str) -> List[Tuple[str, str, str, str]]:
    """Discover SW96 data groups: (data_a, data_b, label, suffix)."""
    pattern = os.path.join(sw96_dir, "data_list_a_SW96_*.npy")
    files = []
    for file_a in sorted(glob.glob(pattern)):
        basename = os.path.basename(file_a)
        m = re.match(r"data_list_a_SW96_(.+)\.npy", basename)
        if not m:
            continue
        suffix = m.group(1)
        file_b = os.path.join(sw96_dir, f"data_list_b_SW96_{suffix}.npy")
        label_file = os.path.join(sw96_dir, f"label_list_SW96_{suffix}.npy")
        if os.path.exists(file_b) and os.path.exists(label_file):
            files.append((file_a, file_b, label_file, suffix))
    return files


def load_sw96_dataset(file_a: str, file_b: str, label_file: str):
    X1 = np.load(file_a)
    X2 = np.load(file_b)
    y = np.load(label_file)
    y_seq1 = y[:, :, 0]
    y_seq2 = y[:, :, 1]
    y_scalar = y[:, 0, 3:5]
    return TensorDataset(
        torch.FloatTensor(X1),
        torch.FloatTensor(X2),
        torch.FloatTensor(y_seq1),
        torch.FloatTensor(y_seq2),
        torch.FloatTensor(y_scalar),
    )


def norm_from_config(config: dict):
    return (
        config.get("range_min", -15068.0),
        config.get("range_max", 15068.0),
        config.get("vel_min", 1.0),
        config.get("vel_max", 5.0),
    )


def list_checkpoints(model_dir: str, best_only: bool, selected_stems: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Return [(checkpoint_path, stem), ...] for the requested checkpoints.

    Args:
        model_dir: Directory containing .pth files.
        best_only: If True, only return best_model.pth.
        selected_stems: If provided, only return checkpoints whose stem is in this list.
    """
    pths = []
    best_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_path):
        pths.append((best_path, "best_model"))
    if not best_only:
        for ckpt in sorted(glob.glob(os.path.join(model_dir, "epoch_*.pth"))):
            stem = os.path.splitext(os.path.basename(ckpt))[0]
            pths.append((ckpt, stem))

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p, s in pths:
        if p not in seen:
            seen.add(p)
            unique.append((p, s))

    if selected_stems is not None:
        selected_set = set(selected_stems)
        unique = [(p, s) for p, s in unique if s in selected_set]

    return unique


def save_figure(fig: plt.Figure, save_path: str, dpi: int = FIGURE_DPI):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Per-model figures (mirroring evaluate_SW96_two_V_R.py)
# -----------------------------------------------------------------------------

def plot_sequence_and_scalar(result: dict, label: str, save_path: str):
    seq1_true = result["seq1_true"]
    seq1_pred = result["seq1_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: two best sequence-1 samples
    mse_per_sample = np.mean((seq1_pred - seq1_true) ** 2, axis=1)
    best_idx = np.argsort(mse_per_sample)[:2]
    colors = ["#1f77b4", "#d62728"]
    for i, idx in enumerate(best_idx):
        axes[0].plot(seq1_true[idx] + 2 * i, label=f"True (sample {idx})", color=colors[i], linestyle="-", linewidth=2)
        axes[0].plot(seq1_pred[idx] + 2 * i + 1, label=f"Predicted (sample {idx})", color=colors[i], linestyle="--", linewidth=2)
    axes[0].set_xlabel("Time step", fontsize=12)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].set_title("Sequence 1 prediction comparison", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: scalar scatter
    axes[1].scatter(result["range_true"], result["range_pred"], alpha=0.5, s=15, color="#2ca02c", label="Range")
    axes[1].scatter(result["vel_true"], result["vel_pred"], alpha=0.5, s=15, color="#d62728", label="Velocity")
    min_v = min(
        result["range_true"].min(), result["range_pred"].min(),
        result["vel_true"].min(), result["vel_pred"].min(),
    )
    max_v = max(
        result["range_true"].max(), result["range_pred"].max(),
        result["vel_true"].max(), result["vel_pred"].max(),
    )
    axes[1].plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.5, linewidth=2, label="Ideal prediction")
    axes[1].set_xlabel("True value", fontsize=12)
    axes[1].set_ylabel("Predicted value", fontsize=12)
    axes[1].set_title("Scalar parameter prediction scatter (range + velocity)", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(label, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, save_path)


def plot_threshold_analysis(result: dict, label: str, save_path: str):
    thresholds = np.arange(0, 1.05, 0.05)
    lc = result["loss_compre"]
    range_errs, vel_errs = [], []
    for t in thresholds:
        mask = lc >= t
        if np.any(mask):
            range_errs.append(np.mean(result["range_err"][mask]))
            vel_errs.append(np.mean(result["vel_err"][mask]))
        else:
            range_errs.append(np.nan)
            vel_errs.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, range_errs, "s-", label="Range error (%)", linewidth=2, markersize=6, color="green")
    ax.plot(thresholds, vel_errs, "^-", label="Velocity error (%)", linewidth=2, markersize=6, color="red")
    ax.set_xlabel("Correlation coefficient threshold", fontsize=12)
    ax.set_ylabel("Mean relative error (%)", fontsize=12)
    ax.set_title(f"Mean prediction error across thresholds\n{label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_sequence_loss_distribution(result: dict, label: str, save_path: str):
    loss_seq1 = np.mean((result["seq1_pred"] - result["seq1_true"]) ** 2, axis=0)
    loss_seq2 = np.mean((result["seq2_pred"] - result["seq2_true"]) ** 2, axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(loss_seq1, "b-", label="Sequence 1 loss", alpha=0.7, linewidth=2)
    ax.plot(loss_seq2, "r-", label="Sequence 2 loss", alpha=0.7, linewidth=2)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Mean squared error", fontsize=12)
    ax.set_title(f"Sequence prediction loss distribution\n{label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_range_result(result: dict, label: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sort_idx = np.argsort(result["range_true"])
    axes[0].scatter(range(len(result["range_true"])), result["range_true"][sort_idx], alpha=0.6, s=10, label="True value", color="green")
    axes[0].scatter(range(len(result["range_pred"])), result["range_pred"][sort_idx], alpha=0.6, s=10, label="Predicted value", color="orange", marker="x")
    axes[0].set_xlabel("Sample index", fontsize=12)
    axes[0].set_ylabel("Range (m)", fontsize=12)
    axes[0].set_title(f"Range prediction results (sorted)\n{label}", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    sorted_lc = result["loss_compre"][np.argsort(result["loss_compre"])]
    sorted_range_err = result["range_err"][np.argsort(result["loss_compre"])]
    axes[1].plot(sorted_lc, sorted_range_err, "r-", alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel("Correlation coefficient", fontsize=12)
    axes[1].set_ylabel("Relative error (%)", fontsize=12)
    axes[1].set_title("Range error vs. correlation coefficient", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_velocity_result(result: dict, label: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sort_idx = np.argsort(result["vel_true"])
    axes[0].scatter(range(len(result["vel_true"])), result["vel_true"][sort_idx], alpha=0.6, s=10, label="True value", color="brown")
    axes[0].scatter(range(len(result["vel_pred"])), result["vel_pred"][sort_idx], alpha=0.6, s=10, label="Predicted value", color="pink", marker="x")
    axes[0].set_xlabel("Sample index", fontsize=12)
    axes[0].set_ylabel("Velocity (knot)", fontsize=12)
    axes[0].set_title(f"Velocity prediction results (sorted)\n{label}", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    sorted_lc = result["loss_compre"][np.argsort(result["loss_compre"])]
    sorted_vel_err = result["vel_err"][np.argsort(result["loss_compre"])]
    axes[1].plot(sorted_lc, sorted_vel_err, "r-", alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel("Correlation coefficient", fontsize=12)
    axes[1].set_ylabel("Relative error (%)", fontsize=12)
    axes[1].set_title("Velocity error vs. correlation coefficient", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_correlation_distribution(result: dict, label: str, save_path: str):
    lc = result["loss_compre"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lc, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(x=np.mean(lc), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(lc):.4f}")
    ax.axvline(x=np.median(lc), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(lc):.4f}")
    for p, c, lbl in zip([25, 50, 75, 90], ["blue", "green", "orange", "red"], ["25th", "50th", "75th", "90th"]):
        val = np.percentile(lc, p)
        ax.axvline(x=val, color=c, linestyle=":", linewidth=1.5, alpha=0.7, label=f"{lbl} percentile: {val:.4f}")
    ax.set_xlabel("Correlation coefficient", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Loss Compre (correlation coefficient) distribution\n{label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_error_vs_correlation(result: dict, label: str, save_path: str):
    lc = result["loss_compre"]
    range_err = result["range_err"]
    vel_err = result["vel_err"]
    sort_idx = np.argsort(lc)
    lc_sorted = lc[sort_idx]
    range_err_sorted = range_err[sort_idx]
    vel_err_sorted = vel_err[sort_idx]

    ws = min(WINDOW_SIZE, max(1, len(lc_sorted) // 2))
    lc_smooth = np.convolve(lc_sorted, np.ones(ws) / ws, mode="valid")
    range_smooth = np.convolve(range_err_sorted, np.ones(ws) / ws, mode="valid")
    vel_smooth = np.convolve(vel_err_sorted, np.ones(ws) / ws, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(lc, range_err, alpha=0.3, s=10, color="green", label="Samples")
    axes[0].plot(lc_smooth, range_smooth, "r-", linewidth=2.5, label="Moving average")
    axes[0].set_xlabel("Correlation coefficient", fontsize=12)
    axes[0].set_ylabel("Relative error (%)", fontsize=12)
    r = stats.pearsonr(lc, range_err)[0]
    axes[0].set_title(f"Range error vs. correlation coefficient\n(Pearson r={r:.4f})", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)

    axes[1].scatter(lc, vel_err, alpha=0.3, s=10, color="saddlebrown", label="Samples")
    axes[1].plot(lc_smooth, vel_smooth, "r-", linewidth=2.5, label="Moving average")
    axes[1].set_xlabel("Correlation coefficient", fontsize=12)
    axes[1].set_ylabel("Relative error (%)", fontsize=12)
    r = stats.pearsonr(lc, vel_err)[0]
    axes[1].set_title(f"Velocity error vs. correlation coefficient\n(Pearson r={r:.4f})", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)

    plt.suptitle(label, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, save_path, dpi=500)


def generate_model_figures(result: dict, config: dict, output_dir: str, label: str):
    """Generate the 7 figures for a single checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    has_seq = config.get("use_physics", True)

    plot_sequence_and_scalar(result, label, os.path.join(output_dir, "01_seq_prediction.png"))
    plot_threshold_analysis(result, label, os.path.join(output_dir, "02_threshold_analysis.png"))
    if has_seq:
        plot_sequence_loss_distribution(result, label, os.path.join(output_dir, "03_seq_loss_distribution.png"))
    plot_range_result(result, label, os.path.join(output_dir, "04_range_result.png"))
    plot_velocity_result(result, label, os.path.join(output_dir, "05_velocity_result.png"))
    if has_seq:
        plot_correlation_distribution(result, label, os.path.join(output_dir, "06_correlation_distribution.png"))
        plot_error_vs_correlation(result, label, os.path.join(output_dir, "07_error_vs_correlation.png"))


# -----------------------------------------------------------------------------
# Cross-variant comparison figures
# -----------------------------------------------------------------------------

def plot_error_comparison(results_dict: dict, save_path: str):
    modes = list(results_dict.keys())
    range_mape = [np.mean(results_dict[m]["range_err"]) for m in modes]
    vel_mape = [np.mean(results_dict[m]["vel_err"]) for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(modes))
    bars1 = axes[0].bar(x, range_mape, 0.6, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Mean relative error (%)", fontsize=12)
    axes[0].set_title("Range prediction error comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([mode_label(m) for m in modes], rotation=30, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)

    bars2 = axes[1].bar(x, vel_mape, 0.6, color="coral", alpha=0.8)
    axes[1].set_ylabel("Mean relative error (%)", fontsize=12)
    axes[1].set_title("Velocity prediction error comparison", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([mode_label(m) for m in modes], rotation=30, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar in bars2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_correlation_complexity(results_dict: dict, save_path: str):
    modes = list(results_dict.keys())
    mean_corr = [np.mean(results_dict[m]["loss_compre"]) for m in modes]
    params = [results_dict[m]["params"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(modes))
    axes[0].bar(x, mean_corr, 0.6, color="mediumpurple", alpha=0.8)
    axes[0].set_ylabel("Mean |ρ|", fontsize=12)
    axes[0].set_title("Sequence correlation coefficient comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([mode_label(m) for m in modes], rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, [p / 1e6 for p in params], 0.6, color="darkorange", alpha=0.8)
    axes[1].set_ylabel("Parameters (M)", fontsize=12)
    axes[1].set_title("Model complexity comparison", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([mode_label(m) for m in modes], rotation=30, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_range_scatter_grid(results_dict: dict, save_path: str):
    modes = list(results_dict.keys())
    n_cols = min(len(modes), 3)
    n_rows = (len(modes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    for idx, mode in enumerate(modes):
        ax = axes[idx // n_cols, idx % n_cols]
        res = results_dict[mode]
        ax.scatter(res["range_true"], res["range_pred"], alpha=0.4, s=10, color="green")
        min_v = min(res["range_true"].min(), res["range_pred"].min())
        max_v = max(res["range_true"].max(), res["range_pred"].max())
        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("True value (m)", fontsize=9)
        ax.set_ylabel("Predicted value (m)", fontsize=9)
        ax.set_title(f"{mode_label(mode)}\nRange MAPE: {np.mean(res['range_err']):.2f}%", fontsize=10)
        ax.grid(True, alpha=0.3)
    for idx in range(len(modes), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_velocity_scatter_grid(results_dict: dict, save_path: str):
    modes = list(results_dict.keys())
    n_cols = min(len(modes), 3)
    n_rows = (len(modes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    for idx, mode in enumerate(modes):
        ax = axes[idx // n_cols, idx % n_cols]
        res = results_dict[mode]
        ax.scatter(res["vel_true"], res["vel_pred"], alpha=0.4, s=10, color="brown")
        min_v = min(res["vel_true"].min(), res["vel_pred"].min())
        max_v = max(res["vel_true"].max(), res["vel_pred"].max())
        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("True value (knot)", fontsize=9)
        ax.set_ylabel("Predicted value (knot)", fontsize=9)
        ax.set_title(f"{mode_label(mode)}\nVelocity MAPE: {np.mean(res['vel_err']):.2f}%", fontsize=10)
        ax.grid(True, alpha=0.3)
    for idx in range(len(modes), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_threshold_comparison(results_dict: dict, save_path: str):
    """
    Plot Range/Velocity error versus correlation threshold.

    Modes whose predicted correlation is essentially constant (e.g. no_physics,
    where |rho| is always 0) are omitted from the line plot because they would
    only produce a flat line. Their threshold-wise data is still preserved in
    the CSV file saved by save_threshold_results_csv().
    """
    thresholds = np.arange(0, 1.05, 0.05)
    # Only plot modes that have meaningful correlation variation.
    modes = [
        m for m in results_dict.keys()
        if np.max(results_dict[m]["loss_compre"]) >= 1e-6
    ]
    if not modes:
        print("  [Warning] No mode has meaningful correlation, skipping threshold comparison plot.")
        return

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(modes)))
    linestyles = ["-", "-.", "--", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, mode in enumerate(modes):
        res = results_dict[mode]
        lc = res["loss_compre"]
        re, ve = [], []
        for t in thresholds:
            mask = lc >= t
            re.append(np.mean(res["range_err"][mask]) if np.any(mask) else np.nan)
            ve.append(np.mean(res["vel_err"][mask]) if np.any(mask) else np.nan)
        ls = linestyles[idx % len(linestyles)]
        label = mode_label(mode)
        ax.plot(thresholds, re, linestyle=ls, marker="o", markevery=5, color=colors[idx], linewidth=2.2, alpha=0.85, label=f"{label} (Range)")
        ax.plot(thresholds, ve, linestyle=ls, marker="^", markevery=5, color=colors[idx], linewidth=2.2, alpha=0.85, label=f"{label} (Velocity)")

    ax.set_xlabel("Correlation coefficient threshold", fontsize=12)
    ax.set_ylabel("Mean relative error (%)", fontsize=12)
    # ax.set_title("Threshold analysis comparison", fontsize=14, fontweight="bold")
    n_entries = 2 * len(modes)
    ax.legend(fontsize=8, ncol=2 if n_entries <= 12 else 3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_path)


def save_threshold_results_csv(results_dict: dict, save_path: str):
    """
    Save threshold-wise Range/Velocity errors and sample counts for all modes.

    This includes modes that are omitted from the threshold line plot (e.g.
    no_physics), so their results are still available for inspection.
    """
    import csv
    thresholds = np.arange(0, 1.05, 0.05)
    rows = []
    for mode in results_dict:
        res = results_dict[mode]
        lc = res["loss_compre"]
        for t in thresholds:
            mask = lc >= t
            if np.any(mask):
                rows.append({
                    "mode": mode,
                    "threshold": round(float(t), 2),
                    "sample_count": int(np.sum(mask)),
                    "range_err(%)": float(np.mean(res["range_err"][mask])),
                    "vel_err(%)": float(np.mean(res["vel_err"][mask])),
                })
            else:
                rows.append({
                    "mode": mode,
                    "threshold": round(float(t), 2),
                    "sample_count": 0,
                    "range_err(%)": "",
                    "vel_err(%)": "",
                })

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "threshold", "sample_count", "range_err(%)", "vel_err(%)"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved: {save_path}")


def generate_comparison_figures(results_dict: dict, output_dir: str, data_name: str):
    """Generate cross-variant comparison figures for one data group."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"{data_name}_comparison")
    plot_error_comparison(results_dict, f"{prefix}_01_error_comparison.png")
    plot_correlation_complexity(results_dict, f"{prefix}_02_correlation_complexity.png")
    plot_range_scatter_grid(results_dict, f"{prefix}_03_range_scatter_grid.png")
    plot_velocity_scatter_grid(results_dict, f"{prefix}_04_velocity_scatter_grid.png")
    plot_threshold_comparison(results_dict, f"{prefix}_05_threshold_comparison.png")
    save_threshold_results_csv(results_dict, f"{prefix}_05_threshold_comparison.csv")
    plot_threshold_sample_distribution(results_dict, f"{prefix}_06_threshold_sample_distribution.png")


def plot_threshold_sample_distribution(results_dict: dict, save_path: str):
    """
    Plot the sample count distribution across correlation coefficient ranges.

    This complements the threshold error comparison by showing how many samples
    fall into each correlation bin for every variant, making it easier to judge
    whether an observed error reduction is supported by a sufficient sample size.

    The no-physics variant is omitted because its predicted correlation is
    always zero and therefore has no meaningful distribution to display.
    """
    modes = [m for m in results_dict.keys() if m != "no_physics"]
    if not modes:
        print("  [Warning] No variants with physics layer, skipping threshold sample distribution plot.")
        return
    # Fixed 5 bins of width 0.2; keeps the grouped bar chart readable with many modes.
    edges = np.arange(0, 1.01, 0.2)
    bin_labels = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]

    counts = {mode: [] for mode in modes}
    for mode in modes:
        lc = results_dict[mode]["loss_compre"]
        for i in range(len(edges) - 1):
            low, high = edges[i], edges[i + 1]
            if i == len(edges) - 2:
                mask = (lc >= low) & (lc <= high)
            else:
                mask = (lc >= low) & (lc < high)
            counts[mode].append(int(np.sum(mask)))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(bin_labels))
    n_modes = len(modes)
    width = 0.7 / max(n_modes, 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_modes))

    for idx, mode in enumerate(modes):
        offset = width * (idx - (n_modes - 1) / 2)
        bars = ax.bar(x + offset, counts[mode], width, label=mode_label(mode), color=colors[idx], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xlabel("Correlation coefficient |ρ| range", fontsize=12)
    ax.set_ylabel("Sample count", fontsize=12)
    # ax.set_title("Sample count distribution across correlation coefficient ranges", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure(fig, save_path)


def save_results_excel(results_dict: dict, save_path: str):
    try:
        import pandas as pd
    except ImportError:
        print("  [Warn] pandas not installed, skipping Excel output")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    summary = []
    for mode, res in results_dict.items():
        summary.append({
            "Variant": mode,
            "Range_MAPE(%)": np.mean(res["range_err"]),
            "Vel_MAPE(%)": np.mean(res["vel_err"]),
            "Mean_|rho|": np.mean(res["loss_compre"]),
            "Parameters": res["params"],
            "Samples": len(res["range_err"]),
        })
    df_summary = pd.DataFrame(summary)

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        for mode, res in results_dict.items():
            n = len(res["range_err"])
            detail = {
                "range_true": res["range_true"],
                "range_pred": res["range_pred"],
                "range_err(%)": res["range_err"],
                "vel_true": res["vel_true"],
                "vel_pred": res["vel_pred"],
                "vel_err(%)": res["vel_err"],
                "|rho|": res["loss_compre"],
            }
            pd.DataFrame(detail).to_excel(writer, sheet_name=mode[:31], index=False)
    print(f"  Excel saved: {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch ablation prediction & plotting on SW96")
    parser.add_argument("--checkpoint_root", required=True, help="Root with ablation_* folders or a single ablation folder")
    parser.add_argument("--sw96_dir", required=True, help="Directory containing prepared SW96 .npy files")
    parser.add_argument("--output_dir", default="predictions", help="Directory for figures and Excel")
    parser.add_argument("--best_only", action="store_true", help="Only evaluate best_model.pth")
    parser.add_argument(
        "--stems",
        nargs="+",
        default=None,
        help=(
            "Checkpoint stems to evaluate, e.g. 'best_model' or "
            "'best_model epoch_0010 epoch_0020'. Default: all available."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    set_seed(42)

    ablation_dirs = discover_ablation_dirs(args.checkpoint_root)
    if not ablation_dirs:
        print(f"No ablation_* directories found under {args.checkpoint_root}")
        return
    print(f"Found {len(ablation_dirs)} ablation directorie(s):")
    for d in ablation_dirs:
        print(f"  - {d}")

    sw96_files = discover_sw96_files(args.sw96_dir)
    if not sw96_files:
        print(f"No SW96 files found in {args.sw96_dir}")
        return
    print(f"Found {len(sw96_files)} SW96 data group(s)")

    os.makedirs(args.output_dir, exist_ok=True)

    for file_a, file_b, label_file, suffix in sw96_files:
        print("\n" + "=" * 70)
        print(f"Data group: SW96_{suffix}")
        print("=" * 70)

        dataset = load_sw96_dataset(file_a, file_b, label_file)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print(f"  Samples: {len(dataset)}")

        # results_by_stem[stem] -> {mode_name: result_dict}
        results_by_stem: Dict[str, Dict[str, dict]] = {}

        for model_dir in ablation_dirs:
            basename = os.path.basename(model_dir)
            # Support both old (MMDD_HHMM) and new (YYYYMMDD_HHMMSS) timestamp formats
            m = re.match(r"^ablation_(.+)_(\d{4,8})_(\d{4,6})$", basename)
            mode_name = m.group(1) if m else basename
            checkpoints = list_checkpoints(model_dir, args.best_only, selected_stems=args.stems)
            if not checkpoints:
                print(f"  [Skip] {model_dir}: no matching .pth found")
                continue

            for ckpt_path, ckpt_stem in checkpoints:
                full_label = f"{mode_name}_{ckpt_stem}"
                print(f"\n  Evaluating {full_label} ...")

                try:
                    config = load_config(model_dir)
                    range_min, range_max, vel_min, vel_max = norm_from_config(config)
                    model = load_model(ckpt_path, config, device)
                    result = evaluate_model(
                        model, loader, device,
                        range_min, range_max, vel_min, vel_max,
                        return_sequences=True,
                    )
                    result["params"] = model.count_parameters()

                    range_mape = np.mean(result["range_err"])
                    vel_mape = np.mean(result["vel_err"])
                    mean_rho = np.mean(result["loss_compre"])
                    print(f"    Range MAPE: {range_mape:.2f}% | Vel MAPE: {vel_mape:.2f}% | |ρ|: {mean_rho:.4f} | Params: {result['params']:,}")

                    # Per-checkpoint figures keep the full label (e.g. full_best_model)
                    per_model_dir = os.path.join(args.output_dir, f"SW96_{suffix}", full_label)
                    generate_model_figures(result, config, per_model_dir, full_label)

                    # Group by checkpoint stem for comparison figures.
                    # Use mode_name (without stem suffix) as the comparison key so labels are clean.
                    results_by_stem.setdefault(ckpt_stem, {})[mode_name] = result

                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"    [Error] {full_label}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if not results_by_stem:
            print("\n  No results to compare.")
            continue

        # Generate one set of comparison figures per checkpoint stem
        for stem, results_dict in results_by_stem.items():
            comparison_dir = os.path.join(args.output_dir, f"SW96_{suffix}", "comparison", stem)
            data_name = f"SW96_{suffix}_{stem}"
            print(f"\n  Generating comparison figures for stem '{stem}' -> {comparison_dir}")
            generate_comparison_figures(results_dict, comparison_dir, data_name)
            excel_path = os.path.join(comparison_dir, f"{data_name}_results.xlsx")
            save_results_excel(results_dict, excel_path)

    print("\nAll predictions and figures completed.")


if __name__ == "__main__":
    main()
