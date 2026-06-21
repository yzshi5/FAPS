# FAPS

Code for **Flow Annealing Posterior Sampling for Function-Space Regression and Inverse Problems**.

Large checkpoints and prepared datasets are stored on Hugging Face:

```text
https://huggingface.co/Yaozhong/FAPS
```

This GitHub repository contains the training/evaluation code and shell scripts.

## Setup

Clone the repository and install the main Python dependencies in your environment:

```bash
git clone https://github.com/yzshi5/FAPS.git
cd FAPS

pip install -U torch numpy scipy matplotlib huggingface_hub torchdiffeq
```

Some MINO/weather experiments also require the MINO model utilities included in this repo.

## Download Artifacts

The scripts expect checkpoints and small prepared test datasets under the repository tree, for example:

```text
PDE_inverse/checkpoints/
PDE_inverse/datasets/
Regression/checkpoints/FAPS_prior/
```

Download the uploaded artifacts from Hugging Face:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Yaozhong/FAPS",
    repo_type="model",
    local_dir=".",
    allow_patterns=[
        "PDE_inverse/checkpoints/**",
        "PDE_inverse/datasets/**",
        "Regression/checkpoints/FAPS_prior/GP_gibbs_epoch_500.pt",
        "Regression/checkpoints/FAPS_prior/GP_matern_epoch_500.pt",
    ],
)
PY
```

For the original PDE train/test data, you can also use:

```bash
cd PDE_inverse
python datasets/download_dataset.py all --output-dir datasets/PDE_inverse
```

To download test files only:

```bash
python datasets/download_dataset.py all --test --output-dir datasets/PDE_inverse
```

## PDE Inverse Problems

All commands below are run from:

```bash
cd PDE_inverse
```

### Train FNO Forward Surrogates

```bash
bash scripts_surrogate/train_darcy_FNO.sh
bash scripts_surrogate/train_poisson_FNO.sh
bash scripts_surrogate/train_helmholtz_FNO.sh
bash scripts_surrogate/train_ns_FNO.sh
```

Evaluate the surrogate:

```bash
bash scripts_surrogate/eval_darcy_FNO.sh
bash scripts_surrogate/eval_poisson_FNO.sh
bash scripts_surrogate/eval_helmholtz_FNO.sh
bash scripts_surrogate/eval_ns_FNO.sh
```

### Train FAPS Priors

FNO prior:

```bash
bash scripts/train_darcy_prior.sh
bash scripts/train_poisson_prior.sh
bash scripts/train_helmholtz_prior.sh
bash scripts/train_ns_prior.sh
```

UNet prior:

```bash
bash scripts/train_darcy_prior_unet.sh
bash scripts/train_poisson_prior_unet.sh
bash scripts/train_helmholtz_prior_unet.sh
bash scripts/train_ns_prior_unet.sh
```

### Run Inverse Evaluation

FNO prior:

```bash
bash scripts/eval_darcy_inverse.sh
bash scripts/eval_poisson_inverse.sh
bash scripts/eval_helmholtz_inverse.sh
bash scripts/eval_ns_inverse.sh
```

UNet prior:

```bash
bash scripts/eval_darcy_inverse_unet.sh
bash scripts/eval_poisson_inverse_unet.sh
bash scripts/eval_helmholtz_inverse_unet.sh
bash scripts/eval_ns_inverse_unet.sh
```

Darcy super-resolution inverse evaluation:

```bash
bash scripts/eval_darcy_inverse_sup.sh
```

### Run All-Test Metrics

FNO prior:

```bash
bash scripts_metrics/eval_darcy_inverse_all_test.sh
bash scripts_metrics/eval_poisson_inverse_all_test.sh
bash scripts_metrics/eval_helmholtz_inverse_all_test.sh
bash scripts_metrics/eval_ns_inverse_all_test.sh
```

UNet prior:

```bash
bash scripts_metrics/eval_darcy_inverse_all_test_unet.sh
bash scripts_metrics/eval_poisson_inverse_all_test_unet.sh
bash scripts_metrics/eval_helmholtz_inverse_all_test_unet.sh
bash scripts_metrics/eval_ns_inverse_all_test_unet.sh
```

PDE inverse outputs are written under:

```text
PDE_inverse/outputs/
```

## Regression Experiments

Run these commands from:

```bash
cd Regression
```

Train priors:

```bash
bash scripts/train_GP_matern_prior.sh
bash scripts/train_GP_gibbs_prior.sh
bash scripts/train_ns_prior.sh
bash scripts/train_bh_prior.sh
bash scripts/train_weather_prior.sh
```

Evaluate regression tasks:

```bash
bash scripts/eval_GP_matern_reg.sh
bash scripts/eval_GP_gibbs_reg.sh
bash scripts/eval_ns_reg.sh
bash scripts/eval_bh_reg.sh
bash scripts/eval_weather_reg.sh
```

Regression outputs are written under:

```text
Regression/outputs/
```

## Notes

- Check shell scripts before long runs; they set GPU devices, checkpoint names, and dataset paths.
- Checkpoints and datasets are intentionally ignored by Git and should be downloaded from Hugging Face.
- If you upload refreshed artifacts to Hugging Face, use:

```bash
bash upload_faps_artifacts.sh
```
