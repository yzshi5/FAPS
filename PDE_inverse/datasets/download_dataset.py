#!/usr/bin/env python3
"""
Script to download PDE datasets from HuggingFace repository.
Repository: jcy20/Remove-old-datasets
"""

import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

# Repository information
REPO_ID = "jcy20/DiffusionPDE-normalized"

# Supported datasets based on the files in the repository
SUPPORTED_DATASETS = ["darcy", "helmholtz", "ns-bounded", "ns-nonbounded", "poisson"]

# File mapping for each dataset
DATASET_FILES = {
    "darcy": {"train": "darcy_hf", "test": "darcy_test_hf"},
    "helmholtz": {"train": "helmholtz_hf", "test": "helmholtz_test_hf"},
    "ns-bounded": {"train": "ns-bounded_hf", "test": "ns-bounded_test_hf"},
    "ns-nonbounded": {"train": "ns-nonbounded_hf", "test": "ns-nonbounded_test_hf"},
    "poisson": {"train": "poisson_hf", "test": "poisson_test_hf"},
}


def download_file(repo_id: str, filename: str, output_dir: Path, repo_type: str = "dataset") -> None:
    """
    Download a single file from HuggingFace repository.

    Args:
        repo_id: HuggingFace repository ID
        filename: Name of the file to download
        output_dir: Directory to save the downloaded file
        repo_type: Type of repository (dataset or model)
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=output_dir,
            force_download=False,
        )
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        raise


def download_dataset(dataset_name: str, output_dir: str, download_train: bool = True) -> None:
    """
    Download a specific dataset.

    Args:
        dataset_name: Name of the dataset to download
        download_train: If True, download train dataset; if False, download test dataset
        output_dir: Directory to save the downloaded files (default: ./datasets/{dataset_name})
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are: {', '.join(SUPPORTED_DATASETS)}")

    # Get the file to download
    dataset_type = "train" if download_train else "test"
    dir_name = DATASET_FILES[dataset_name][dataset_type]

    print(f"Downloading {dataset_name} {dataset_type} dataset...")

    output_dir = Path(output_dir)
    repo_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    for file in repo_files:
        if file.startswith(dir_name):
            download_file(REPO_ID, file, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download PDE datasets from HuggingFace repository.",
        epilog=f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=f"Name of the dataset ({', '.join(SUPPORTED_DATASETS)}) or 'all' to download all datasets",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Download the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./datasets/PDE_inverse",
        help="Output directory for downloaded files",
    )

    args = parser.parse_args()

    # Download datasets
    if args.dataset.lower() == "all":
        for dataset in SUPPORTED_DATASETS:
            print(f"\n{'=' * 60}")
            download_dataset(dataset, args.output_dir, download_train=not args.test)
    else:
        download_dataset(args.dataset, args.output_dir, download_train=not args.test)

    print("\n" + "=" * 60)
    print("Finished processing all requested files.")

    # Remove cache
    shutil.rmtree(Path(args.output_dir) / ".cache")
    print("Cache directory removed.")


if __name__ == "__main__":
    main()