"""Extract 8,000/1,000/1,000 original records using only the standard library.

Preview the selection without writing files:
    python amazon_data_10k/prepare_10k.py --dry-run

Reproduce the selection in a new directory and create its .tar.gz archive:
    python amazon_data_10k/prepare_10k.py --output amazon_data_10k_reproduced

Existing outputs are never overwritten. Compressed .pklz files are copied
unchanged, retaining images, dimensions, masks, weights, and IDs.
"""

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import tarfile
from pathlib import Path


BASE = Path(__file__).resolve().parent
SPLITS = (("train", "train_data", 8000), ("dev", "val_data", 1000), ("test", "test_data", 1000))


def select_records(source, seed=42):
    """Match the original sampling order: sorted files, one RNG, train/dev/test."""
    rng = random.Random(seed)
    selected, source_counts = [], {}
    for project_split, source_split, count in SPLITS:
        files = sorted((source / source_split).glob("*.pklz"), key=lambda path: path.name)
        if len(files) < count:
            raise ValueError(f"{source_split}: need {count} records, found {len(files)}")
        source_counts[source_split] = len(files)
        selected.extend((path, project_split, source_split) for path in rng.sample(files, count))
    return selected, source_counts


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=BASE.parent / "amazon_data")
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show counts without creating or changing files")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    selected, source_counts = select_records(source, args.seed)
    if args.dry_run:
        print(json.dumps({"seed": args.seed, "source_counts": source_counts,
                          "selected_counts": {split: count for split, _, count in SPLITS}}, indent=2))
        return
    archive = output.with_name(output.name + ".tar.gz")
    if output == source or source in output.parents:
        parser.error("Output must be outside the original data directory")
    if output.exists() or archive.exists():
        parser.error("Output already exists. Use --dry-run or choose a new --output directory")

    for split, _, _ in SPLITS:
        (output / split).mkdir(parents=True, exist_ok=False)
    rows = []
    for path, project_split, source_split in selected:
        target = output / project_split / path.name
        source_hash = sha256(path)
        shutil.copyfile(path, target)
        if sha256(target) != source_hash:
            raise RuntimeError(f"Copy checksum mismatch: {target}")
        rows.append({"project_split": project_split, "source_split": source_split,
                     "id": int(path.name.split("_", 1)[0]),
                     "source_path": path.relative_to(source).as_posix(),
                     "subset_path": target.relative_to(output).as_posix(),
                     "file_sha256": source_hash})
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sampling = {"seed": args.seed, "python_version": platform.python_version(),
                "selection_method": "uniform random sampling without replacement within each original split",
                "selection_implementation": "one random.Random(seed); source filenames sorted lexicographically; train, dev, test sampled in that order",
                "source_counts": source_counts,
                "project_counts": {split: count for split, _, count in SPLITS},
                "original_field_values_preserved": True, "source_split_assignment_preserved": True}
    (output / "sampling.json").write_text(json.dumps(sampling, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__).resolve(), output / "prepare_10k.py")
    (output / "README.md").write_text(
        "# image2mass 10,000-record subset\n\n"
        "Inputs: RGB image, physical dimensions, and supplied thickness mask. Target: weight in pounds. All original record fields are retained.\n\n"
        f"Seed: {args.seed}. Uniform sampling without replacement: train 8,000, dev 1,000, test 1,000, preserving original split assignments.\n\n"
        "manifest.csv lists selected records and checksums. sampling.json records the sampling settings. prepare_10k.py contains the extraction code and requires Python 3.11 or later, with no third-party dependencies.\n\n"
        "To reproduce from inside this folder, with the full source download in ../amazon_data:\n\n"
        f"```sh\npython prepare_10k.py --source ../amazon_data --output ../amazon_data_10k_reproduced --seed {args.seed}\n```\n\n"
        "Use --dry-run to preview the selection. Existing outputs are never overwritten.\n",
        encoding="utf-8")
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(output, arcname=output.name)
    print(f"Saved {len(rows):,} records to {output}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
