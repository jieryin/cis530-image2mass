# Qwen3-VL-2B pointwise fine-tuning results

Seed 42, three epochs, selected full-run epoch 3 (update 2,607). Final test pair accuracy is **89.5% (895/1,000)**; the predefined volume-rule-wrong subset scores **56.5% (95/168)**. Model selection used dev results before test evaluation.

- [Analysis and reproducibility](analysis.md)
- [All test pairs, weights and predictions](test_cases.csv)
- [Final test metrics](test/scores.json) and [hard-subset metrics](test/subset_scores.json)
- [Data verification](data_verification.json), [checkpoint recovery evidence](recovery.json), and [training history](full/history.jsonl)
- [Artifact metadata and checksums](artifacts.json)

## Complete downloadable artifacts

The readable files and scripts are included here with their original paths. Binary model artifacts are available in the [experiment release](https://github.com/jieryin/cis530-image2mass/releases/tag/pointwise-qwen3-vl-2b-seed42-20261125):

- [Complete results ZIP](https://github.com/jieryin/cis530-image2mass/releases/download/pointwise-qwen3-vl-2b-seed42-20261125/pointwise-qwen3-vl-2b-seed42-results.zip) — includes all run files, best/last inference adapters with mass-head tensors, processors and four resumable checkpoints, including optimizer/scheduler/RNG state.
- [Artifact file manifest](https://github.com/jieryin/cis530-image2mass/releases/download/pointwise-qwen3-vl-2b-seed42-20261125/artifact_manifest.csv) — relative paths, byte sizes and SHA256 for every archived file.
- [SHA256SUMS](https://github.com/jieryin/cis530-image2mass/releases/download/pointwise-qwen3-vl-2b-seed42-20261125/SHA256SUMS) — checksums for the ZIP and manifest.

The archive excludes only caches, locks and pending/temporary files. It does not duplicate the base model or dataset; their fixed revision and verified checksums are in the analysis. Extract the ZIP and resume the selected run through full/latest_checkpoint.json using the recorded source, model and dataset. best/ and last/ are inference artifacts.
