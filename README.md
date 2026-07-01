# Correlation-Aware Dual-Tonal LSTM-Transformer for Underwater Target Motion Inversion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of the paper:

> **A Correlation-Aware Dual-Tonal LSTM-Transformer Network for Motion Parameter Inversion of Underwater Targets in Shallow-Water**
>
> Zhuo Chen, Dazhi Gao*, Kai Sun, Yueqi Yu, Ziwen Wang, Longfei Li  
> Department of Marine Technology, Ocean University of China

## Highlights

- **End-to-end regression** of initial slant range and velocity from dual-tonal complex pressure.
- **Physics-informed design**: Pearson correlation of predicted tonal power spectra is embedded in the network and used as a reliability indicator.
- **Dual-branch LSTM-Transformer** hybrid encoder captures both local temporal dynamics and long-range interference dependencies.
- **Gated residual fusion** dynamically balances acoustic-field features and physics-informed correlation features.
- Trained purely on simulated data and validated on the **SWellEx-96 (SW96)** real-world experiment without fine-tuning.

## Repository Structure

```text
.
├── train_ablation.py                  # Main entry: train full model & all ablation variants
├── predict_ablation.py                # Batch evaluation & visualization on SW96 data
├── model.py                           # Neural network architectures (full + baselines)
├── dataset.py                         # Simulation data loading / train-val-test split
├── losses.py                          # Multi-task loss with correlation constraints
├── configs/
│   ├── config.yaml                    # Example config for the full model
│   └── ablation_run.yaml              # Config used for the ablation study
├── matlab/
│   ├── make_data_to_bin_par_fenkuai_paris_diff.m   # Simulation data generation (KRAKEN-based)
│   ├── generateFreqPairs.m                         # Frequency-pair generation helper
│   ├── merged_script.m                             # Baseline method + SW96 preprocessing demo
│   └── merged_script_1.m                           # FDM velocity + dual-line spectrum figures
├── requirements.txt
├── LICENSE
└── README.md
```

## Environment

- Python >= 3.8
- PyTorch >= 2.0 (CUDA 11.8 or 12.1 recommended)
- MATLAB R2020a or later (for data generation scripts)
- See `requirements.txt` for full Python dependencies.

```bash
conda create -n underwater python=3.10
conda activate underwater
pip install -r requirements.txt
```

## Data

### 1. Simulation Data (Training / Validation)

The simulation data is generated in MATLAB using KRAKEN normal-mode outputs. The Python dataloader expects `.npy` files under the `data_path` configured in the YAML:

```text
<data_path>/
├── env_{01..11}_data_list_a_{01..05}_0005e3.npy   # main input   (N, 400, 5)
├── env_{01..11}_data_list_b_{01..05}_0005e3.npy   # auxiliary input (N, 1400, 2)
├── env_{01..11}_label_list_{01..05}_0005e3.npy    # labels       (N, 400, 5)
└── ...
```

- Environments 1–11 are used for training/validation.
- Environment 12 is the held-out test set.

**How to generate**: Run `matlab/make_data_to_bin_par_fenkuai_paris_diff.m` after preparing the KRAKEN eigenvalue/eigenfunction files (`k*.txt`, `phi*.txt`) and `SSP_data_12.mat`. The script performs the following steps:

1. Loads the sound-speed profile (SSP) for each environment.
2. Randomly samples source/receiver geometry, velocity, CPA range, and CPA time.
3. Synthesizes dual-frequency pressure fields via normal-mode summation.
4. Adds AWGN noise, estimates the waveguide-invariant `beta`, and normalizes inputs/labels.
5. Saves the processed tensors as `.npy` files using `writeNPY`.

**MATLAB dependencies for data generation**:

- `generateFreqPairs.m` (included in this repo).
- `writeNPY` from [npy-matlab](https://github.com/kwikteam/npy-matlab) (please clone/add to MATLAB path).
- MATLAB Parallel Computing Toolbox (`parpool`).
- MATLAB Signal Processing Toolbox (`awgn`).

**External inputs required by the MATLAB scripts** (not included in this repo due to size):

- KRAKEN normal-mode eigenvalues and eigenfunctions: `karkenc/txt_{env_number}/k{freq}.txt` and `phi{freq}.txt`.
- Sound-speed profiles: `SSP_data_12.mat`.

### 2. SW96 Real Data (Validation)

The prepared SW96 `.npy` files are expected under `--sw96_dir`:

```text
<sw96_dir>/
├── data_list_a_SW96_{suffix}.npy
├── data_list_b_SW96_{suffix}.npy
└── label_list_SW96_{suffix}.npy
```

These files are produced by preprocessing the original SWellEx-96 Event S5 acoustic recordings and GPS trajectory. See `matlab/merged_script.m` for a reference preprocessing pipeline that includes:

- GPS time-base conversion and great-circle distance calculation.
- Acoustic spectrogram computation (`spectrogram`).
- Sliding-window segment extraction excluding the CPA region.
- Normalization and `.npy` export.

**MATLAB dependencies for SW96 preprocessing**:

- MATLAB Mapping Toolbox (`distance`, `deg2km`).
- SIO reader (`sioread`) for `.sio` acoustic files.

## Quick Start

### 1. Train the Full Model

```bash
python train_ablation.py --config configs/config.yaml --modes full
```

### 2. Train All Ablation Variants

```bash
python train_ablation.py --config configs/ablation_run.yaml --modes all
```

Or train a custom subset:

```bash
python train_ablation.py --config configs/ablation_run.yaml \
  --modes full no_gated_fusion no_physics lstm_only transformer_only \
  --save_root checkpoints/ablation_run
```

Each variant is saved under `checkpoints/ablation_batch_{timestamp}/ablation_{mode}_{timestamp}/` containing:

- `config.yaml` / `config.json` – merged configuration.
- `model_config.json`, `data_config.json`, `train_config.json` – split legacy configs (used by some plotting scripts).
- `best_model.pth` – best validation checkpoint.
- `epoch_{NNNN}.pth` – periodic checkpoints every 10 epochs (configurable via `save_every`).

### 3. Evaluate on SW96

```bash
python predict_ablation.py \
  --checkpoint_root checkpoints/ablation_run \
  --sw96_dir "/path/to/SW96_npy" \
  --output_dir predictions/ablation_run
```

To evaluate only the best checkpoint:

```bash
python predict_ablation.py \
  --checkpoint_root checkpoints/ablation_run \
  --sw96_dir "/path/to/SW96_npy" \
  --output_dir predictions/ablation_run \
  --best_only
```

Output structure:

```text
predictions/ablation_run/
└── SW96_{suffix}/
    ├── full_best_model/                      # per-checkpoint figures
    │   ├── 01_seq_prediction.png
    │   ├── 02_threshold_analysis.png
    │   ├── 03_seq_loss_distribution.png
    │   ├── 04_range_result.png
    │   ├── 05_velocity_result.png
    │   ├── 06_correlation_distribution.png
    │   └── 07_error_vs_correlation.png
    ├── no_physics_best_model/
    └── comparison/
        └── best_model/
            ├── SW96_{suffix}_best_model_comparison_01_error_comparison.png
            ├── SW96_{suffix}_best_model_comparison_02_correlation_complexity.png
            ├── SW96_{suffix}_best_model_comparison_03_range_scatter_grid.png
            ├── SW96_{suffix}_best_model_comparison_04_velocity_scatter_grid.png
            ├── SW96_{suffix}_best_model_comparison_05_threshold_comparison.png
            ├── SW96_{suffix}_best_model_comparison_05_threshold_comparison.csv
            ├── SW96_{suffix}_best_model_comparison_06_threshold_sample_distribution.png
            └── SW96_{suffix}_best_model_results.xlsx
```

## Model Architecture

The proposed network consists of four modules:

1. **Dual-branch hybrid encoder**  
   - Main branch (5 ch × 400): frequency, complex pressure magnitude, SSP.
   - Auxiliary branch (2 ch × 1400): phase-normalized squared field difference.
2. **LSTM-Transformer encoder**  
   Bidirectional LSTM + multi-head self-attention with positional encoding and residual connections.
3. **Physics-informed correlation layer**  
   Predicts dual-tonal power spectra, computes Pearson correlation `ρ`, and embeds it as a high-dimensional feature.
4. **Gated residual fusion + scalar head**  
   Dynamically fuses global acoustic features with correlation features and outputs initial slant range and velocity.

## Results

At a correlation threshold of `|ρ| = 0.9`:

| Dataset | Initial Slant Range MAPE |
|---------|--------------------------|
| Simulation | ~5.8% |
| SW96 real data | ~17.4% (down from ~103.6% for all samples) |

See the paper for full experimental details.

## Known Limitations & Missing Components

While the core training/inference pipeline and data-generation MATLAB scripts are included, the following components are **not** shipped with this repository and must be prepared or implemented by the user:

1. **KRAKEN normal-mode outputs**  
   The eigenvalue/eigenfunction files (`k*.txt`, `phi*.txt`) and `SSP_data_12.mat` are generated externally by the [KRAKEN](https://oalib-acoustics.org/) normal-mode model.

2. **Original SW96 raw data**  
   The raw `.sio` acoustic files and Event S5 GPS trajectories must be downloaded from the [SWellEx-96 repository](https://sio.ucsd.edu/px/SCS/SWellEx-96/).

3. **`writeNPY` MATLAB function**  
   Required by the MATLAB data-generation script. Install from [npy-matlab](https://github.com/kwikteam/npy-matlab).

4. **Dedicated `prepare_sw96.py`**  
   Currently SW96 preprocessing is done in MATLAB (`merged_script.m`). A Python-only preprocessing script is a future convenience addition.

5. **Standalone `eval_sim.py` / `eval_real.py`**  
   The current unified evaluation entry point is `predict_ablation.py`. Separate lightweight evaluators may be added later.

6. **`benchmark.py`**  
   A formal model-complexity benchmark (FLOPs / latency / memory) is not yet included.

7. **Pre-trained weights**  
   Model checkpoints are excluded from git (see `.gitignore`). If you would like to share trained weights, upload them to a separate cloud storage (Google Drive / Baidu Netdisk / Hugging Face) and link them in the repository release notes.

## File Upload Checklist

The following files have been included in this repository:

### Required Python source
- [x] `train_ablation.py`
- [x] `predict_ablation.py`
- [x] `model.py`
- [x] `dataset.py`
- [x] `losses.py`
- [ ] `evaluate.py` *(not yet included; shared evaluation utilities will be added in a future update)*
- [x] `configs/config.yaml`
- [x] `configs/ablation_run.yaml`
- [x] `requirements.txt`

### Optional / supplementary Python source *(not yet included)*
- [ ] `analyze_physical_overfitting.py`
- [ ] `fig_paper_architecture.py`
- [ ] `fig_paper_unified_no_scatter.py`

### MATLAB source (under `matlab/`)
- [x] `matlab/make_data_to_bin_par_fenkuai_paris_diff.m`
- [x] `matlab/generateFreqPairs.m`
- [x] `matlab/merged_script.m`
- [x] `matlab/merged_script_1.m`

### Repository metadata
- [x] `README.md`
- [x] `LICENSE`
- [x] `.gitignore`

### Should be excluded
- [ ] `__pycache__/` *(git-ignored)*
- [ ] `checkpoints/` *(large model weights, git-ignored)*
- [ ] `predictions/` *(generated figures/CSVs, git-ignored)*
- [ ] `data/` *(large `.npy` / `.mat` files, git-ignored)*
- [ ] `*.log` *(git-ignored)*

## Citation

If you use this code, please cite:

```bibtex
@article{chen2025correlation,
  title={A Correlation-Aware Dual-Tonal LSTM-Transformer Network for Motion Parameter Inversion of Underwater Targets in Shallow-Water},
  author={Chen, Zhuo and Gao, Dazhi and Sun, Kai and Yu, Yueqi and Wang, Ziwen and Li, Longfei},
  journal={Engineering Applications of Artificial Intelligence},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- KRAKEN normal-mode propagation model: <https://oalib-acoustics.org/>
- SWellEx-96 experiment data: <https://sio.ucsd.edu/px/SCS/SWellEx-96/>
- `writeNPY` for MATLAB: <https://github.com/kwikteam/npy-matlab>
