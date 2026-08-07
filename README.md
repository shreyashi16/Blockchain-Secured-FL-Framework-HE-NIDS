# Blockchain-Secured Federated Learning Framework with Homomorphic Encryption for Network Intrusion Detection

**Scope of this repository**: everything between *Dataset Upload* and *Global XGBoost Model* in the
overall system pipeline - i.e. **Federated Learning + Homomorphic Encryption + XGBoost integration**
only. Blockchain, the dashboard/UI, and all data preprocessing are explicitly **out of scope** and
owned by other teammates.

```
CSV Flow Data → Preprocessing (other team) → Dataset Upload → [THIS REPO: Federated Learning
across 3 Organizations → Homomorphic Encryption of Updates → Secure Aggregation → Global XGBoost
Model] → Blockchain Verification (future) → Dashboard Visualization (future)
```

---

## 1. Dataset

Source: **CIC-IDS2018** flow records (generated with CICFlowMeter), already fully preprocessed by
the data team (missing values handled, encoded, scaled, class-balanced). This repo makes **no**
changes to preprocessing - it only reads the CSVs.

| File | Rows | Columns | Role |
|---|---|---|---|
| `dataset/client_1_X.csv` / `client_1_y.csv` | 17,226 | 70 features | Organization A local data |
| `dataset/client_2_X.csv` / `client_2_y.csv` | 17,226 | 70 features | Organization B local data |
| `dataset/client_3_X.csv` / `client_3_y.csv` | 17,226 | 70 features | Organization C local data |
| `dataset/X_test.csv` / `y_test.csv` | 7,621 | 70 features | Shared, held-out **global** test set |

* **Label column**: `Label` - already integer-encoded, **13 classes** (`0`-`12`) covering Benign and
  multiple attack categories (DDoS, Brute Force, Botnet, Infiltration, Web Attack, etc.).
* **Features**: 70 numeric, already standardized (e.g. `Dst Port`, `Flow Duration`,
  `Tot Fwd Pkts`, `Flow Byts/s`, `Fwd IAT Mean`, `SYN Flag Cnt`, `Active Mean`, `Idle Std`, ...
  the full flow-based CICFlowMeter feature set). No categorical/text columns remain.
* Each client file already represents one organization's isolated local dataset. `X_test`/`y_test`
  is a single dataset shared only for **evaluating the global model** - it is never used for local
  client training.
* Per-client data has no dedicated test split of its own, so each client carves out a local
  validation split (`config.LOCAL_VALIDATION_SPLIT`, default 15%) from its own training data purely
  for local round-by-round evaluation - this is the "train-test split performed within the FL
  pipeline" the spec calls for when one is absent.

If you later swap in a **single combined** preprocessed CSV instead of pre-split client files, set
`config.PARTITION_STRATEGY = "iid"` or `"noniid"` and point `config.COMBINED_DATASET_PATH` at it;
`federated/dataset_partition.py` will partition it across `config.NUM_CLIENTS` clients automatically
(IID = uniform random split, Non-IID = Dirichlet label-skew split).

---

## 2. Architecture

```
project/
├── config.py                  # every hyperparameter / path, single source of truth
├── train.py                   # entry point: runs full FL + HE training
├── evaluate.py                # loads a saved global model, evaluates on the test set
├── requirements.txt
├── README.md
├── dataset/                   # preprocessed CSVs (as uploaded)
├── logs/                      # structured logs (created at runtime)
├── outputs/
│   ├── plots/                 # all required visualizations (png)
│   ├── models/                # persisted global model bundle
│   └── metadata/              # round history + blockchain-ready metadata (json)
├── federated/
│   ├── dataset_partition.py   # loading + IID/non-IID/preassigned partitioning
│   ├── client.py              # local training / evaluation / update generation
│   ├── aggregation.py         # FedAvg weighting + weighted-ensemble global model
│   ├── server.py              # round orchestration, evaluation, broadcast
│   └── trainer.py             # top-level multi-round orchestrator
├── encryption/
│   ├── context.py             # CKKS context generation (TenSEAL)
│   ├── keys.py                # public/secret key distribution
│   ├── encrypt.py             # encrypts client update vectors, hashing for blockchain
│   ├── decrypt.py             # decrypts + validates aggregated result
│   └── secure_aggregation.py  # full per-round HE pipeline
├── models/
│   └── xgboost_model.py       # XGBoost wrapper: binary/multiclass, class balancing, warm-start
└── notebook_demo.ipynb
```

### 2.1 Federated Learning

* **3 organizations** (`config.NUM_CLIENTS = 3`) by default, one per `client_i_X/y.csv`.
* Each communication round, every participating client (`config.PARTICIPATION_RATIO` controls the
  fraction sampled) trains its local `XGBoostModel` for `config.LOCAL_EPOCHS` additional boosting
  rounds, **warm-started** from its own model of the previous round (so "local epochs" genuinely
  accumulate learning across rounds rather than retraining from scratch).
* Local evaluation (on the client's own validation split) happens every round and is logged and
  plotted.
* **Data never leaves a client.** Only a compact numeric *update vector* - the client's normalized
  gain-based feature-importance vector concatenated with its local accuracy and macro-F1 - is
  produced for aggregation, and that vector is what gets homomorphically encrypted.

### 2.2 Global model aggregation

Gradient-boosted tree leaf values cannot be meaningfully averaged element-wise the way neural
network weights can (different clients' trees have different structures). This project therefore
uses the standard, well-established strategy for federated tree ensembles: the **global model is a
sample-count-weighted soft-voting ensemble** of participating clients' boosters -

```
P_global(y | x) = Σ_c  w_c · P_client_c(y | x),      w_c ∝ local sample count  (Σ w_c = 1)
```

This is FedAvg's weighting rule applied at the probability level, and it is well-defined regardless
of tree structure/count. See the module docstring in `federated/aggregation.py` for the full
rationale.

### 2.3 Homomorphic Encryption (TenSEAL / CKKS)

The numeric update vector each client produces (feature importances + local accuracy/F1) **is**
what gets protected under HE and is what the server aggregates without ever seeing an individual
client's plaintext update:

1. `encryption/context.py` — generates a CKKS context (`poly_modulus_degree=8192`,
   `coeff_mod_bit_sizes=[60,40,40,60]`). A secret-key-stripped **public** context is what clients
   receive; the secret key is confined to the coordinator performing decryption.
2. `encryption/keys.py` — distributes the public key material, exposes the decryption context
   separately.
3. `encryption/encrypt.py` — each client encrypts its update vector into a `CKKSVector` and computes
   a SHA-256 hash of the serialized ciphertext (this hash is what a blockchain module would anchor
   on-chain).
4. `encryption/secure_aggregation.py` — the server computes the **weighted sum of ciphertexts
   homomorphically** (`Σ w_c · Enc(update_c)`), i.e. it aggregates without decrypting any individual
   update.
5. `encryption/decrypt.py` — only the final **aggregate** is decrypted, then validated against the
   plaintext-computed weighted sum (CKKS is an approximate scheme, so validation checks closeness
   within `config.HE_CONFIG.validation_atol`, not bit-exactness). Every round's console/log output
   reports `Validation PASSED/FAILED` with the max/mean absolute error.

**Trust model note** (also documented in `encryption/context.py`): this is the standard
"honest-but-curious single aggregator" simplification. A production multi-organization deployment
would replace the single secret key with threshold/multi-key HE so no single party can decrypt
anything alone - that upgrade only touches `encryption/context.py` and `encryption/keys.py`.

### 2.4 XGBoost integration

* `models/xgboost_model.py` auto-detects the number of classes from the local label array and
  configures `objective="binary:logistic"` or `"multi:softprob"` accordingly (this dataset is
  multiclass with 13 classes).
* Class imbalance is handled at **training time only** (never touching preprocessing):
  `scale_pos_weight` for binary tasks, per-sample `compute_sample_weight(class_weight="balanced")`
  for multiclass tasks (toggle via `config.XGBConfig.use_class_balancing`).
* Supports warm-start continuation (`xgb_model=...`) so local training genuinely continues across
  communication rounds.

### 2.5 Blockchain-ready interface

`encryption/secure_aggregation.py` returns a `BlockchainMetadataRecord` per client per round with
exactly: `round_num`, `client_id`, `encrypted_update_hash`, `ciphertext_size_bytes`,
`aggregation_weight`, `model_version`. `train.py` serializes every round's records to
`outputs/metadata/round_history.json` - the blockchain team can consume this file (or the dataclass
directly) without any change to the FL/HE code.

---

## 3. Evaluation & Visualizations

Computed every round, for both each local client and the global model: **accuracy, macro
precision/recall/F1, ROC-AUC (OvR macro for multiclass), log-loss, confusion matrix**
(`utils/metrics.py`).

Plots auto-saved to `outputs/plots/` (`utils/visualization.py`):

1. `round_vs_accuracy.png` — Communication Round vs Accuracy (global + per-client)
2. `round_vs_f1.png` — Communication Round vs F1 Score (global + per-client)
3. `round_vs_loss.png` — Communication Round vs Loss
4. `client_comparison_final_round.png` — Client-wise performance comparison (bar chart)
5. `global_roc_curve_final_round.png` — ROC Curve (macro-average one-vs-rest)
6. `global_confusion_matrix_final_round.png` — Confusion Matrix (normalized heatmap)

---

## 4. Logging

Structured logs (timestamp, level, module, message) written to `logs/`:

* `logs/main.log` — everything, one combined stream
* `logs/training.log` — dataset partitioning, client/trainer local-training logs
* `logs/encryption.log` — context/key generation, per-update encryption, decryption + validation
* `logs/aggregation.log` — FedAvg weight computation, per-round server orchestration

---

## 5. Configuration

All hyperparameters live in `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `NUM_CLIENTS` | `3` | Number of simulated organizations |
| `COMMUNICATION_ROUNDS` | `10` | Number of FL rounds |
| `LOCAL_EPOCHS` | `20` | New boosting rounds trained locally per FL round |
| `PARTICIPATION_RATIO` | `1.0` | Fraction of clients sampled each round |
| `PARTITION_STRATEGY` | `"preassigned"` | `"preassigned"` \| `"iid"` \| `"noniid"` |
| `LOCAL_VALIDATION_SPLIT` | `0.15` | Local train/val split per client |
| `RANDOM_SEED` | `42` | Global reproducibility seed |
| `AGGREGATION_WEIGHTING` | `"sample_count"` | `"sample_count"` \| `"uniform"` |
| `HE_CONFIG` | CKKS, `poly_modulus_degree=8192` | Homomorphic encryption parameters |
| `XGB_CONFIG` | see `config.py` | XGBoost hyperparameters |

---

## 6. Setup & Execution

```bash
# 1. Create an environment
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full federated training pipeline
python train.py
#   optional overrides:
python train.py --rounds 15 --local-epochs 30 --participation-ratio 0.67 --partition noniid

# 4. Evaluate a saved global model on the held-out test set
python evaluate.py
```

Outputs after `train.py`:

* `logs/*.log` — full structured logs
* `outputs/plots/*.png` — all six required visualizations
* `outputs/metadata/round_history.json` — per-round metrics + blockchain metadata
* `outputs/models/global_model.pkl` — the persisted global ensemble (client boosters + weights)

### Notebook demo

`notebook_demo.ipynb` walks through the same pipeline interactively (a short 3-round run) with
inline plots - useful for a live demo/screenshot walkthrough. Launch with:

```bash
jupyter notebook notebook_demo.ipynb
```

---

## 7. Explicitly out of scope

Per the project brief, this repository does **not** implement:

* Blockchain recording/verification (consumes `BlockchainMetadataRecord` / `round_history.json` as
  a clean interface instead)
* Dashboard / UI
* Data preprocessing, cleaning, feature engineering, encoding, scaling, or class-balancing of the
  raw dataset (the uploaded CSVs are assumed correct and are consumed as-is)
* CICFlowMeter / raw flow generation
