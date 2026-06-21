#!/usr/bin/env python3
"""
Convert Hugging Face dataset Arrow shards into one .npy file per dataset folder.

Input example:
    /oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse/darcy_hf/data-00000-of-00027.arrow

Output example:
    /oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy/darcy_flow.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "pyarrow is required for conversion. Install it with: pip install pyarrow"
    ) from exc


DEFAULT_INPUT_ROOT = Path("./datasets/PDE_inverse")
DEFAULT_OUTPUT_ROOT = Path("./datasets/PDE_inverse_npy")


def discover_dataset_dirs(root: Path) -> list[Path]:
    """Return subdirectories that contain Arrow shard files."""
    return sorted(
        path for path in root.iterdir() if path.is_dir() and any(path.glob("*.arrow"))
    )


def sorted_arrow_files(dataset_dir: Path) -> list[Path]:
    """Return Arrow files sorted by shard index."""
    return sorted(dataset_dir.glob("*.arrow"))


def output_name(dataset_dir_name: str) -> str:
    """
    Map folder names to readable output names.
    Example: darcy_hf -> darcy_flow.npy
    """
    if dataset_dir_name.endswith("_test_hf"):
        base = dataset_dir_name[: -len("_test_hf")].replace("-", "_")
        return f"{base}_flow_test.npy"
    if dataset_dir_name.endswith("_hf"):
        base = dataset_dir_name[: -len("_hf")].replace("-", "_")
        return f"{base}_flow.npy"
    return f"{dataset_dir_name.replace('-', '_')}.npy"


def load_dataset_info(dataset_dir: Path) -> dict:
    info_file = dataset_dir / "dataset_info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {info_file}")
    return json.loads(info_file.read_text())


def pick_data_column(dataset_info: dict, preferred_column: str) -> str:
    features = dataset_info.get("features", {})
    if preferred_column in features:
        return preferred_column
    for name in features:
        if name != "id":
            return name
    raise ValueError("Could not infer a data column from dataset_info.json")


def read_arrow_column(file_path: Path, column_name: str) -> np.ndarray:
    """Read one Arrow shard and return the selected column as a dense NumPy array."""
    with pa.memory_map(str(file_path), "r") as source:
        try:
            table = ipc.RecordBatchFileReader(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            table = ipc.RecordBatchStreamReader(source).read_all()
    if column_name not in table.column_names:
        raise KeyError(f"Column '{column_name}' not found in {file_path.name}")
    return np.asarray(table[column_name].to_pylist())


def iterate_shards(dataset_dir: Path) -> Iterator[Path]:
    for shard in sorted_arrow_files(dataset_dir):
        yield shard


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
    data_column: str,
    dtype: np.dtype | None = None,
) -> Path:
    """Convert one dataset folder into one .npy file."""
    info = load_dataset_info(dataset_dir)
    column = pick_data_column(info, data_column)
    num_examples = info["splits"]["train"]["num_examples"]
    feature_shape = tuple(info["features"][column]["shape"])
    np_dtype = np.dtype(dtype or info["features"][column]["dtype"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name(dataset_dir.name)

    target = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np_dtype,
        shape=(num_examples, *feature_shape),
    )

    cursor = 0
    for shard in iterate_shards(dataset_dir):
        shard_array = read_arrow_column(shard, column).astype(np_dtype, copy=False)
        n_rows = shard_array.shape[0]
        target[cursor : cursor + n_rows] = shard_array
        cursor += n_rows
        print(
            f"[{dataset_dir.name}] merged {shard.name} ({n_rows} rows) -> "
            f"{cursor}/{num_examples}"
        )

    if cursor != num_examples:
        raise RuntimeError(
            f"Row mismatch for {dataset_dir.name}: expected {num_examples}, got {cursor}"
        )

    target.flush()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all Arrow shard folders into one .npy per folder "
            "under a new output root."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root containing dataset folders (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root to write .npy files (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--data-column",
        type=str,
        default="data",
        help="Column name to export. Falls back to first non-id feature if missing.",
    )
    args = parser.parse_args()

    dataset_dirs = discover_dataset_dirs(args.input_root)
    if not dataset_dirs:
        raise SystemExit(f"No dataset folders with .arrow files found in {args.input_root}")

    print(f"Found {len(dataset_dirs)} dataset folders in {args.input_root}")
    for dataset_dir in dataset_dirs:
        print("=" * 80)
        print(f"Converting {dataset_dir} ...")
        out_file = convert_dataset(
            dataset_dir=dataset_dir,
            output_dir=args.output_root,
            data_column=args.data_column,
        )
        print(f"Saved: {out_file}")

    print("=" * 80)
    print("All dataset folders converted to .npy successfully.")


if __name__ == "__main__":
    main()
