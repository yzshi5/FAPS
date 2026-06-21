#!/usr/bin/env python3
"""Upload FAPS checkpoint and dataset folders to Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = os.environ.get("HF_REPO_ID", "Yaozhong/FAPS")
REPO_TYPE = os.environ.get("HF_REPO_TYPE", "model")
ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "PDE_inverse/checkpoints",
    "PDE_inverse/datasets",
]

FILES = [
    "Regression/checkpoints/FAPS_prior/GP_gibbs_epoch_500.pt",
    "Regression/checkpoints/FAPS_prior/GP_matern_epoch_500.pt",
]

IGNORE_PATTERNS = [
    "**/.cache/**",
    "**/__pycache__/**",
    "**/.DS_Store",
    "**/*.sh",
    "**/*.csv",
]


def main() -> None:
    api = HfApi()

    for folder in FOLDERS:
        folder_path = ROOT / folder
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder_path}")
    for file in FILES:
        file_path = ROOT / file
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing file: {file_path}")

    print(f"Uploading artifacts to {REPO_ID} (repo type: {REPO_TYPE})")

    for folder in FOLDERS:
        folder_path = ROOT / folder
        print(f"\nUploading {folder_path} -> {folder}")
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=folder_path,
            path_in_repo=folder,
            commit_message=f"Upload {folder}",
            ignore_patterns=IGNORE_PATTERNS,
        )

    for file in FILES:
        file_path = ROOT / file
        print(f"\nUploading {file_path} -> {file}")
        api.upload_file(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_or_fileobj=file_path,
            path_in_repo=file,
            commit_message=f"Upload {file}",
        )

    print("\nUpload complete.")


if __name__ == "__main__":
    main()
