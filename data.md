# Data: Object Mass Estimation

## Source and overview

We use the Amazon product dataset released with Standley et al. (2017), _image2mass: Estimating the Mass of an Object from Its Image_, presented at CoRL. The authors collected approximately 3 million Amazon product listings and filtered them to approximately 150,000 examples containing product images, weights, and dimensions. See the [paper](https://proceedings.mlr.press/v78/standley17a/standley17a.pdf) and [official project repository](https://github.com/tstandley/image2mass).

Our downloaded release contains 149,914 records: 141,950 training, 6,529 validation, and 1,435 test records. It covers varied consumer products, including household and kitchen items, toys, tools, and electronics/accessories. Examples inspected include pens, cups, bowls, stuffed toys, baking dishes, and cables. In the full downloaded release, weights range from 0.01 lb (0.00454 kg) to 14 lb (6.35 kg), with a median of 0.9 lb (0.408 kg); the middle 50% lie between 0.35 lb (0.159 kg) and 2.5 lb (1.134 kg). These statistics describe the full source data, rather than our selected subset.

## Sampling and splits

Full original download: [amazon_data.zip (Google Drive)](https://drive.google.com/file/d/1EnL_sSfZBgxUiL63W6UHWIsp00-rID_A/view?usp=drive_link).

Our project dataset contains 10,000 records randomly selected, sampled within the original splits to preserve their separation.

| Project split | Source split | Records | Percentage |
|---|---|---:|---:|
| Training | Original training | 8,000 | 80% |
| Development | Original validation | 1,000 | 10% |
| Test | Original test | 1,000 | 10% |
| Total | | 10,000 | 100% |

Each selected record belongs to exactly one split. Development data supports parameter tuning and model selection; test data is reserved for final evaluation.

Sampling uses seed `42`. The extraction code is included in `amazon_data_10k/prepare_10k.py`; it sorts filenames before sampling and preserves the original records without modification. `manifest.csv` lists the selected records, and `sampling.json` records the sampling settings.

## File format and record structure

Each record is stored as a `.pklz` file containing an LZ4-compressed Python pickle dictionary. The image is embedded in the record.

| Field | Original type and meaning | Project use |
|---|---|---|
| `image_data` | Bytes containing a 512 × 512 RGB WebP image | Model input, after image decoding |
| `weight` | String containing the weight in pounds | Numeric mass target, after conversion to `float` |
| `id` | Integer record identifier | Tracking only |
| `dimensions` | Tuple of three strings describing size in inches | Model input |
| `thickness_mask` | 32 × 32 NumPy `float32` array of estimated thickness | Model input |

For example, the original record `101262_bowl wooden.pklz` shows a wooden bowl and has the following structure. Large byte and array values are abbreviated:

```python
{
    "dimensions": ("6.00", "6.00", "3.00"),
    "image_data": b"...",       # 13,600 encoded WebP bytes; INPUT
    "id": 101262,
    "thickness_mask": ...,     # NumPy array: shape (32, 32), dtype float32
    "weight": "0.25"           # TARGET: 0.25 lb (0.1134 kg)
}
```

For model fine-tuning, we use `image_data`, `dimensions`, and `thickness_mask` as inputs and `weight` as the target. The `id` field is used for tracking only.

## Data submission and download

The selected original records are packaged as `amazon_data_10k.tar.gz`, with this directory structure:

```text
amazon_data_10k/
├── train/    # 8,000 .pklz records
├── dev/      # 1,000 .pklz records
├── test/     # 1,000 .pklz records
├── prepare_10k.py
├── manifest.csv
├── sampling.json
└── README.md
```

The included script requires Python 3.11 or later and uses only the standard library. With the full original `amazon_data/` directory available, the selection can be reproduced in a new directory:

```sh
python amazon_data_10k/prepare_10k.py --source amazon_data --output amazon_data_10k_reproduced --seed 42
```

Full 10,000-record project dataset: [Download amazon_data_10k.tar.gz (GitHub Release)](https://github.com/jieryin/cis530-image2mass/releases/download/v1.0.0/amazon_data_10k.tar.gz).
