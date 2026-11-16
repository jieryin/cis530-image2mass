# CIS530: Object Mass Estimation

## Environment setup

Python 3.11 is the supported project version. Create and activate a virtual environment, then install the required dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Pairwise mass comparison task

Project documentation, milestone materials, and code for a reproducible 10,000-record subset of the Amazon image2mass dataset.

| Split | Original source | Records |
|---|---|---:|
| train | train_data | 8,000 |
| dev | val_data | 1,000 |
| test | test_data | 1,000 |

Inputs are the RGB image, physical dimensions, and supplied thickness mask. The target is the listed weight. Sampling uses seed 42 and preserves the original split assignments and record bytes.

The generated pair files are stored in the `pairs/` directory:

- `pairs/train_pairs.jsonl`
- `pairs/dev_pairs.jsonl`
- `pairs/test_pairs.jsonl`


## Downloads

- Project 10k subset: [amazon_data_10k.tar.gz](https://github.com/jieryin/cis530-image2mass/releases/download/v1.0.0/amazon_data_10k.tar.gz).
- Full original download: [amazon_data.zip on Google Drive](https://drive.google.com/file/d/1EnL_sSfZBgxUiL63W6UHWIsp00-rID_A/view?usp=drive_link).

The 10k archive is distributed as a Release asset rather than committed to Git. The full original data stays on Google Drive.

## Files

- `data.md`: dataset description and input fields.
- `CIS530_Milestone1.pdf`: milestone submission.
- `amazon_data_10k/prepare_10k.py`: extraction code.
- `amazon_data_10k/manifest.csv`: selected records and checksums.
- `amazon_data_10k/sampling.json`: sampling parameters and statistics.
- `amazon_data_10k/README.md`: extraction instructions.
- `SOURCE_DATA_TERMS.txt`: terms supplied with the original dataset.

## Reproduce the subset

Place the original source download in `amazon_data/`, then run with Python 3.11 or later:

```sh
python amazon_data_10k/prepare_10k.py --source amazon_data --output amazon_data_10k_reproduced --seed 42
```

The script uses only the Python standard library. Add `--dry-run` to preview the selection. Existing outputs are never overwritten.

## Run the simple baseline

The simple baseline supports two methods:

- `majority`: predicts the most frequent A/B label in the training set for every test example.
- `volume`: predicts that the object with the larger bounding-box volume is heavier.

Run the training-majority baseline from the repository root:

```bash
python simple_baseline/simple-baseline.py \
  --method majority \
  --train pairs/train_pairs.jsonl \
  --test pairs/test_pairs.jsonl \
  --output simple_baseline/simple_preds.jsonl
```

To run the volume baseline:

```bash
python simple_baseline/simple-baseline.py \
  --method volume \
  --train pairs/train_pairs.jsonl \
  --test pairs/test_pairs.jsonl \
  --output simple_baseline/volume_preds.jsonl
```

## Run the evaluation script

Score the majority-baseline predictions from the repository root:

```bash
python score.py \
  pairs/test_pairs.jsonl \
  simple_baseline/simple_preds.jsonl
```

The primary evaluation metric is accuracy. The scorer also reports macro-F1, per-class precision, recall and F1, a confusion matrix, and prediction coverage.

For the complete evaluation protocol and file-format requirements, see [`scoring.md`](scoring.md).


## Source

Standley et al. (2017), [image2mass](https://github.com/tstandley/image2mass). The original dataset's terms remain applicable; see `SOURCE_DATA_TERMS.txt`.
