"""
dataset.py
==========
Data loading utilities for the proposed model.

Expected saved data format (produced by generate_data.py / preprocess.py):
    env_{ee}_data_list_a_{ff}_{nnn}.npy   : main input   (N, 400, 5)
    env_{ee}_data_list_b_{ff}_{nnn}.npy   : auxiliary input (N, 1400, 2)
    env_{ee}_label_list_{ff}_{nnn}.npy    : labels       (N, 400, 4)

Label channels:
    0: beta-scaled power spectrum
    1: f2 power spectrum
    2: initial slant range (signed)
    3: velocity
"""

import os
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_environment_data(
    data_path: str,
    env_range: Tuple[int, int] = (1, 12),
    file_range: Tuple[int, int] = (1, 6),
    seq_lens: Tuple[int, int] = (400, 1400),
    input_sizes: Tuple[int, int] = (5, 2),
    sample_num_str: str = "0005e3",
    single_spectrum: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Load simulation data from multiple environments/files.

    Returns concatenated tensors:
        X1, X2, y_seq1, y_seq2, y_scalar
    """
    X1_list, X2_list = [], []
    y_seq1_list, y_seq2_list, y_scalar_list = [], [], []

    for env_one in range(env_range[0], env_range[1]):
        for file_one in range(file_range[0], file_range[1]):
            suffix = f"{file_one:02d}_{sample_num_str}.npy"
            x1_file = os.path.join(
                data_path, f"env_{env_one:02d}_data_list_a_{suffix}"
            )
            x2_file = os.path.join(
                data_path, f"env_{env_one:02d}_data_list_b_{suffix}"
            )
            y_file = os.path.join(data_path, f"env_{env_one:02d}_label_list_{suffix}")

            if not all(os.path.exists(f) for f in [x1_file, x2_file, y_file]):
                continue

            X1_temp = np.load(x1_file)
            X2_temp = np.load(x2_file)
            y_temp = np.load(y_file)

            X1_list.append(X1_temp)
            X2_list.append(X2_temp)
            y_seq1_list.append(y_temp[:, :, 0])
            y_seq2_list.append(y_temp[:, :, 1])
            # Existing saved labels have 5 channels; scalar targets are at indices 3 and 4.
            # If your labels only have 4 channels, change this to y_temp[:, 0, 2:4].
            y_scalar_list.append(y_temp[:, 0, 3:5])

    if not X1_list:
        raise ValueError(f"No data files found in {data_path} with given ranges.")

    X1 = np.concatenate(X1_list, axis=0)
    X2 = np.concatenate(X2_list, axis=0)
    y_seq1 = np.concatenate(y_seq1_list, axis=0)
    y_seq2 = np.concatenate(y_seq2_list, axis=0)
    y_scalar = np.concatenate(y_scalar_list, axis=0)

    # Single-spectrum ablation
    if single_spectrum:
        X1 = X1[:, :, [0, 2, 4]]
        X2 = X2[:, :, [0]]

    return (
        torch.FloatTensor(X1),
        torch.FloatTensor(X2),
        torch.FloatTensor(y_seq1),
        torch.FloatTensor(y_seq2),
        torch.FloatTensor(y_scalar),
    )


def build_dataloaders(
    data_path: str,
    seq_lens: Tuple[int, int] = (400, 1400),
    input_sizes: Tuple[int, int] = (5, 2),
    env_range: Tuple[int, int] = (1, 12),
    file_range: Tuple[int, int] = (1, 6),
    sample_num_str: str = "0005e3",
    test_size: float = 0.1,
    val_size: float = 0.1,
    batch_size: int = 512,
    num_workers: int = 4,
    single_spectrum: bool = False,
    seed: int = 42,
):
    """Build train/val/test DataLoaders."""
    X1, X2, y_seq1, y_seq2, y_scalar = load_environment_data(
        data_path=data_path,
        env_range=env_range,
        file_range=file_range,
        seq_lens=seq_lens,
        input_sizes=input_sizes,
        sample_num_str=sample_num_str,
        single_spectrum=single_spectrum,
    )

    indices = np.arange(X1.shape[0])
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        random_state=seed,
    )

    train_ds = TensorDataset(
        X1[train_idx], X2[train_idx], y_seq1[train_idx], y_seq2[train_idx], y_scalar[train_idx]
    )
    val_ds = TensorDataset(
        X1[val_idx], X2[val_idx], y_seq1[val_idx], y_seq2[val_idx], y_scalar[val_idx]
    )
    test_ds = TensorDataset(
        X1[test_idx], X2[test_idx], y_seq1[test_idx], y_seq2[test_idx], y_scalar[test_idx]
    )

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(num_workers // 2, 0),
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(num_workers // 2, 0),
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
