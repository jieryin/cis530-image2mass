# image2mass 10,000-record project subset

This subset follows the current project data.md: RGB image + physical dimensions + supplied thickness mask as inputs; weight as target.

| Project split | Source split | Objects |
|---|---|---:|
| train | train_data | 8,000 |
| dev | val_data | 1,000 |
| test | test_data | 1,000 |

Uniform sampling without replacement, seed 42. Source filenames are sorted; one local Python random.Random instance samples train, dev, test in order.

All original records are preserved byte-for-byte with all five fields. No similarity filtering, deduplication, relabeling, resizing, or field removal was applied.

manifest.csv records original/project split, ID, paths, weight, dimensions, and checksums. sampling.json records the procedure and actual subset statistics.

Weights are in pounds; physical dimensions are in inches. Thickness masks are supplied estimates, not measured depth. Weight is a target, not a prediction-time input; IDs and filenames are for tracking only.

The included `prepare_10k.py` contains the extraction code. It requires Python 3.11 or later and uses only the standard library. To reproduce from inside this folder, with the full source download in `../amazon_data`:

```sh
python prepare_10k.py --source ../amazon_data --output ../amazon_data_10k_reproduced --seed 42
```

Add `--dry-run` to preview the selection without writing files. Existing outputs are never overwritten.

Source: Standley et al. (2017), https://github.com/tstandley/image2mass . The source dataset's original terms remain applicable; see amazon_data/README in the source download.
