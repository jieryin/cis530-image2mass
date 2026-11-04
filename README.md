# CIS530: Object Mass Estimation

Project documentation, milestone materials, and code for a reproducible 10,000-record subset of the Amazon image2mass dataset.

| Split | Original source | Records |
|---|---|---:|
| train | train_data | 8,000 |
| dev | val_data | 1,000 |
| test | test_data | 1,000 |

Inputs are the RGB image, physical dimensions, and supplied thickness mask. The target is the listed weight. Sampling uses seed 42 and preserves the original split assignments and record bytes.

## Downloads

- Project 10k subset: [amazon_data_10k.tar.gz](https://github.com/jieryin/cis530-image2mass/releases/download/untagged-b0834176464a154e3c3c/amazon_data_10k.tar.gz).
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

## Collaboration

The repository owner can invite group members through **Settings > Collaborators > Add people**. After accepting the invitation, collaborators can submit changes. Use a separate branch and pull request when reviewing each other's work.

## Source

Standley et al. (2017), [image2mass](https://github.com/tstandley/image2mass). The original dataset's terms remain applicable; see `SOURCE_DATA_TERMS.txt`.
