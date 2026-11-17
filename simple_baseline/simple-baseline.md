# Simple baselines

`simple-baseline.py` implements two baselines that use no learning from images.

**Majority class** (`--method majority`, the default) predicts the most frequent label in the training pairs, B, for every pair. Since the A/B order of each pair is random, it scores at chance.

**Volume rule** (`--method volume`)predicts that the object with the larger volume is heavier:

$$
\hat{y} =
\begin{cases}
A & \text{if } d_{a,1} d_{a,2} d_{a,3} > d_{b,1} d_{b,2} d_{b,3} \\
B & \text{if } d_{b,1} d_{b,2} d_{b,3} > d_{a,1} d_{a,2} d_{a,3}
\end{cases}
$$

where $d_{x,1}, d_{x,2}, d_{x,3}$ are the three listed dimensions of object $x$ in inches. The pairs this rule gets wrong are those where the larger object is lighter, which is where a model must use visual cues such as material.

## Input and output

The inputs are the pair files in `pairs/`, produced by `make_pairs.py`, one JSON object per line:

```json
{"pair_id": "test-00000", "id_a": 6852, "id_b": 7125, "path_a": "...", "path_b": "...", "dims_a": [4.13, 2.76, 1.57], "dims_b": [6.69, 2.76, 1.65], "weight_a": 0.04, "weight_b": 0.15, "label": "B"}
```

The output is one prediction per evaluation pair, in the format read by `score.py`:

```json
{"pair_id": "test-00000", "pred": "B"}
```

## Usage

From the repository root:

```sh
python simple_baseline/simple-baseline.py --train pairs/train_pairs.jsonl --test pairs/test_pairs.jsonl --output simple_baseline/simple_preds.jsonl
python simple_baseline/simple-baseline.py --method volume --train pairs/train_pairs.jsonl --test pairs/test_pairs.jsonl --output simple_baseline/volume_preds.jsonl
python score.py pairs/test_pairs.jsonl simple_baseline/simple_preds.jsonl
python score.py pairs/test_pairs.jsonl simple_baseline/volume_preds.jsonl
```

The last two commands score the predictions with our evaluation script.

## Sample output

```text
$ python simple_baseline/simple-baseline.py --method volume --train pairs/train_pairs.jsonl --test pairs/test_pairs.jsonl --output simple_baseline/volume_preds.jsonl
method: volume; majority label: B; volume ties: 0; wrote 1000 predictions to simple_baseline/volume_preds.jsonl

$ head -3 simple_baseline/volume_preds.jsonl
{"pair_id": "test-00000", "pred": "B"}
{"pair_id": "test-00001", "pred": "B"}
{"pair_id": "test-00002", "pred": "B"}

$ python score.py pairs/test_pairs.jsonl simple_baseline/volume_preds.jsonl
Examples:  1000
Correct:   832
Coverage:  1.0000
Accuracy:  0.8320
Macro-F1:  0.8320
...
```

## Score

Each split has 1,000 pairs. The majority label in the 8,000 training pairs is `B`. Scored with `score.py`:

| Baseline | Dev accuracy | Test accuracy | 
|---|---:|---:|---:|---:|
| Majority class | 0.5040| 0.4930 | 
| Volume rule | 0.8210 | 0.8320 | 
