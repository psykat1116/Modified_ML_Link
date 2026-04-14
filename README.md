# ML-Link + Temporal Side Information on the CNS Dataset

This README describes how to reproduce all six experiments (E1–E4, E7–E8) from the
project *"Temporal Side Information for Multilayer Link Prediction"* on the
**Copenhagen Networks Study (CNS)** dataset.

> Drive Link [https://drive.google.com/drive/folders/1uSUb8uHL9hCG8NQGk6DR4rqsYfPhHuVY](https://drive.google.com/drive/folders/1uSUb8uHL9hCG8NQGk6DR4rqsYfPhHuVY)

---

## 1. Project Overview

The base model is **ML-Link** — a multilayer link prediction framework that combines
a GNN branch (ML-GAT on a supra-adjacency matrix) with a structural feature branch
(ISL/ESL/CLA). This project adds **temporal side information** learned from the CNS
interaction layers (calls, SMS, Bluetooth) to improve prediction of Facebook friendships.

The CNS dataset has four layers:

| Layer | Index | Description | Role |
|-------|-------|-------------|------|
| Calls | 0 | Voice calls between students | Context |
| SMS | 1 | Text messages | Context |
| Bluetooth | 2 | Physical proximity | Context |
| Facebook Friends | 3 | Facebook friendship graph | **Target** |

Each experiment removes a fraction of `fb_friends` edges for testing, trains on the
remainder, and evaluates using AUC and Average Precision (AP).

---

## 3. Requirements & Installation
### Python version

Python 3.8 or 3.9 is recommended. Python 3.10+ may require adjustments for some
PyTorch-Geometric packages.

### Install dependencies
```bash
pip install torch==1.13.1
pip install dgl==1.1.1 -f https://data.dgl.ai/wheels/repo.html
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install networkx==3.0 numpy==1.24.1 pandas==1.5.3 \
            scikit-learn==1.3.0 scipy==1.11.1 tqdm==4.64.1
```

Or install all at once from `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **GPU note:** The commands above install CUDA 11.6 wheels. Adjust the `+cu116`
> suffix to match your CUDA version, or use `+cpu` for CPU-only.

### Verify installation
```python
python -c "import torch, dgl; print(torch.__version__, dgl.__version__)"
# Expected: 1.13.1  1.1.1
```

---

## 4. Data Preparation
### Step 1 — Obtain the raw CNS data
Download the Copenhagen Networks Study dataset from
[figshare](https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433).
You need the following files:

- `bt_symmetric.csv` — Bluetooth proximity events
- `calls.csv` — phone call records
- `sms.csv` — SMS records
- `facebook_friends.csv` — Facebook friendship list

Place them in `data/cns_raw/`.

### Step 2 — Build `net.edges` and `meta_info.txt`
The CNS loader (`input_data/cns_load.py`) expects the network in ML-Link's standard
format inside `data/nets/cns/`. Create `meta_info.txt`:

```
N    L    E
692  4    UNDIRECTED
```

Build `net.edges` with one edge per line in the format `<layer> <src> <dst>`:

- Layer 1 = calls, Layer 2 = SMS, Layer 3 = Bluetooth, Layer 4 = fb_friends
- Node IDs must be integers starting from 0
- No duplicate edges; for undirected graphs, list each edge once

The provided `edges.csv` (columns: `source, target, timestamp`) and `nodes.csv`
can be used to construct `net.edges`. Example conversion:

```python
import pandas as pd

edges = pd.read_csv("data/cns_raw/edges.csv", comment="#",
                    names=["layer", "src", "dst", "ts"])
# edges already in layer/src/dst format — drop timestamp column
edges[["layer", "src", "dst"]].drop_duplicates().to_csv(
    "data/nets/cns/net.edges", sep=" ", index=False, header=False)
```

### Step 3 — Build temporal features

The temporal encoder needs per-node activity tensors of shape `(N, T=7, 9)`.
Generate them by running:

```bash
python temporal_preprocess.py \
    --src data/cns_raw \
    --dst data/cns_temporal
```

This reads `bt_symmetric.csv`, `calls.csv`, and `sms.csv` from `data/cns_raw/`,
divides the study period into **T = 7 windows of ~4 days** each, and saves:

| File | Contents |
|------|----------|
| `data/cns_temporal/node_features.pkl` | Dict `{node_id: array(T,9)}` — 9 activity features per window |
| `data/cns_temporal/context_adj.npz` | Union adjacency of calls+SMS+BT (used for hard negative sampling) |
| `data/cns_temporal/degree_arrays.pkl` | Per-layer degree arrays used by the encoder |

**Arguments for `temporal_preprocess.py`:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--src` | `data/cns_raw` | Directory with raw CNS CSV files |
| `--dst` | `data/cns_temporal` | Output directory for temporal tensors |

---

## 5. Running the Experiments

All experiment scripts perform a sweep over removal rates `{10%, 20%, …, 90%}` with
`n_runs=5` independent trials per rate, and save results to CSV + JSON in `artifacts/`.

---

### E1 — Baseline ML-Link

Standard ML-Link with identity matrix node features and random negative sampling.

```bash
python cns_experiment.py \
    --prep_dir data/nets \
    --gpu 0 \
    --n_runs 5 \
    --epochs 100
```

**Output:** `artifacts/cns_removal_results.csv` and `artifacts/cns_removal_results.json`
---

### E2 — k-Hop Extension (k=3)

Extends ML-Link's structural branch to use k-hop neighborhoods instead of the
default 1-hop. The GNN-NE branch is unaffected — it always uses the original
1-hop supra-adjacency.

#### How k-hop adjacency is computed

The k-hop adjacency is built once per trial from the 1-hop training graphs using
matrix powers:

```
A_k = sign(A + 1/2·A² + 1/4·A³ + ... + 1/2^(k-1)·Aᵏ)
```

Self-loops are removed and the result is binarized. It is then converted to a
bidirectional DGL graph with edge weight `w = 1.0` for compatibility with `sfg.py`.
The per-layer edge counts are printed to stdout on each run, for example:

```
[k-hop] layer 0: 1-hop edges=348    3-hop edges=12,451
[k-hop] layer 1: 1-hop edges=348    3-hop edges=11,837
[k-hop] layer 2: 1-hop edges=79,530 3-hop edges=79,530
[k-hop] layer 3: 1-hop edges=3,851  3-hop edges=48,203
```

> Layer 2 (Bluetooth) is already 33% dense — k-hop adds no new edges there.
> The structural gain comes primarily from the sparse call and SMS layers.

#### Run command

```bash
python cns_experiment_khop.py \
    --prep_dir data/nets \
    --gpu 0 \
    --n_runs 5 \
    --epochs 100 \
    --k_hop 3
```

**Output:** `artifacts/cns_removal_results_k3.csv` and `artifacts/cns_removal_results_k3.json`

The output CSV includes an additional leading `k_hop` column:

```
k_hop, removal_rate, auc_mean, auc_std, ap_mean, ap_std
3,     0.1,          94.79,    4.00,    95.19,   3.67
3,     0.2,          94.66,    3.98,    94.97,   3.76
...
```

**All arguments (inherits all E1 arguments, plus):**

| Argument | Default | Description |
|----------|---------|-------------|
| `--k_hop` | `1` | Structural neighborhood radius. `1` = original 1-hop ML-Link behavior. Recommended values: `3` or `5` |

> **Note on k=1:** When `--k_hop 1` is passed, `precompute_khop_graphs()` returns
> the original training graphs unchanged, so the script is equivalent to E1.

#### Trying other k values

The output filename is named automatically based on the k value:

```bash
python cns_experiment_khop.py --prep_dir data/nets --gpu 0 --k_hop 1
# → artifacts/cns_removal_results_k1.csv

python cns_experiment_khop.py --prep_dir data/nets --gpu 0 --k_hop 3
# → artifacts/cns_removal_results_k3.csv

python cns_experiment_khop.py --prep_dir data/nets --gpu 0 --k_hop 5
# → artifacts/cns_removal_results_k5.csv
```

#### Ablation: disabling individual branches

Both branch flags are available in the k-hop script:

```bash
# Structural branch only (disables GNN-NE)
python cns_experiment_khop.py --prep_dir data/nets --k_hop 3 --no_gnn

# GNN-NE branch only (disables structural features)
python cns_experiment_khop.py --prep_dir data/nets --k_hop 3 --no_struct
```

#### Checkpoint naming

Each trial saves its best checkpoint as:
```
checkpoint/{dataset}_k{k_hop}_rate{removal_rate:.2f}_run{run_idx}.pt
```
For example: `checkpoint/cns_k3_rate0.40_run2.pt`.
Checkpoints are deleted automatically at the end of each trial.

---

### E3 — Temporal v1 (Random Negatives)

Temporal encoder (Time2Vec + Transformer) combined with ML-Link. Negative samples
are drawn uniformly from all non-fb pairs — results are inflated due to easy negatives
and are included only as a reference point.

```bash
python cns_experiment_temporal.py \
    --prep_dir data/nets \
    --temp_dir data/cns_temporal \
    --gpu 0 \
    --n_runs 5
```

**Output:** `artifacts/cns_temporal_results.csv`

To reproduce the fixed-seed variant (seed=42):

```bash
python cns_experiment_temporal.py \
    --prep_dir data/nets \
    --temp_dir data/cns_temporal \
    --gpu 0 \
    --n_runs 5 \
    --seed 42
```

**Output:** `artifacts/cns_temporal_results_42.csv`

---

### E4 — Temporal v2 (Relative Features)

Uses window-to-window delta features (changes in activity between consecutive
windows) instead of absolute activity levels. Underperforms the baseline.

```bash
python cns_experiment_temporal.py \
    --prep_dir data/nets \
    --temp_dir data/cns_temporal \
    --gpu 0 \
    --n_runs 5 \
    --feature_mode relative
```

**Output:** `artifacts/cns_temporal_v2_results.csv`

---

### E7 — Temporal Side Info

Temporal embeddings are used as the sole node features for ML-Link's GNN. Negative samples are still random; results are inflated and are included as an ablation reference.

```bash
python cns_experiment_temporal.py \
    --prep_dir data/nets \
    --temp_dir data/cns_temporal \
    --gpu 0 \
    --n_runs 5
```

**Output:** `artifacts/cns_temporal_v5_results.csv`
---

### E8 — Final: Hard Negatives + Side Information

The definitive experiment. Temporal embeddings are concatenated with the identity
matrix, and **hard negative sampling** ensures negatives are structurally and
temporally non-trivial.

#### What are hard negatives?

Hard negatives are node pairs that have real BT/SMS/calls contact but are *not*
Facebook friends. They are sampled by `_sample_hard_negatives()` in `cns_load.py`:

1. Build the context union adjacency: `A_ctx = A_calls ∪ A_SMS ∪ A_BT`
2. Hard pool = pairs in `A_ctx` but not in `A_fb_friends`
3. Sample the required number of negatives from the hard pool
4. If the hard pool is insufficient, fill the remainder with random non-fb pairs

This prevents trivial separation between positives (some temporal activity) and
random negatives (zero temporal activity) that inflated AUC in E3–E7.

#### Run command

```bash
# Step 1: build temporal features (skip if already done in Section 4)
python temporal_preprocess.py \
    --src data/cns_raw \
    --dst data/cns_temporal

# Step 2: run the final experiment
python cns_experiment_temporal_side.py \
    --prep_dir data/nets \
    --temp_dir data/cns_temporal \
    --gpu 0 \
    --n_runs 5 \
    --epochs 100
```

**Output:** `artifacts/cns_temporal_side_results.csv` and `artifacts/cns_temporal_side_results.json`

---

## 6. Understanding the Output Files

Each experiment writes two files to `artifacts/`:

### CSV file

```
removal_rate, auc_mean, auc_std, ap_mean, ap_std
0.1,          93.05,    5.69,    93.01,   5.70
0.2,          96.85,    4.98,    96.80,   5.19
...
```

The k-hop experiment (E2) adds a leading `k_hop` column:

```
k_hop, removal_rate, auc_mean, auc_std, ap_mean, ap_std
3,     0.1,          94.79,    4.00,    95.19,   3.67
```

### JSON file

Includes per-run AUC and AP values for full reproducibility:

```json
[
  {
    "removal_rate": 0.1,
    "auc_mean": 93.05,
    "auc_std":  5.69,
    "ap_mean":  93.01,
    "ap_std":   5.70,
    "auc_runs": [87.43, 89.58, 99.93, 88.29, 99.99],
    "ap_runs":  [87.84, 89.43, 99.92, 87.87, 99.99]
  },
  ...
]
```

### Output file naming

| Experiment | CSV filename |
|------------|-------------|
| E1 Baseline | `cns_removal_results.csv` |
| E2 k-hop k=N | `cns_removal_results_kN.csv` |
| E3 Temporal v1 (random seed) | `cns_temporal_results.csv` |
| E3 Temporal v1 (seed=42) | `cns_temporal_results_42.csv` |
| E4 Temporal v2 (relative) | `cns_temporal_v2_results.csv` |
| E7 Side info, no identity | `cns_temporal_v5_results.csv` |
| E8 Final (hard neg) | `cns_temporal_side_results.csv` |

---

## Quick Reference — All Commands

```bash
python temporal_preprocess.py --src data/cns_raw --dst data/cns_temporal

python cns_experiment.py \
    --prep_dir data/nets --gpu 0 --n_runs 5 --epochs 100

python cns_experiment_khop.py \
    --prep_dir data/nets --gpu 0 --n_runs 5 --epochs 100 --k_hop 3

python cns_experiment_temporal.py \
    --prep_dir data/nets --temp_dir data/cns_temporal --gpu 0 --n_runs 5

python cns_experiment_temporal.py \
    --prep_dir data/nets --temp_dir data/cns_temporal --gpu 0 --n_runs 5 --seed 42

python cns_experiment_temporal_side.py \
    --prep_dir data/nets --temp_dir data/cns_temporal --gpu 0 --n_runs 5 --epochs 100
```
