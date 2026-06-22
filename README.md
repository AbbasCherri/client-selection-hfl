# client-selection-hfl

Simulation code for **"Reputation and Utility-Aware Client Selection for
Hierarchical Multi-Modal Federated Learning in Post-Earthquake Building
Damage Classification"**.

## Key change: streaming data — no download required

The dataset is now **streamed on-demand from HuggingFace** using the
`datasets` library.  Aerial building chips are fetched lazily from the
Japan GSI tile API and cached locally in `./data/tile_cache/`.
You no longer need to download the ~128 k-building snapshot before running.

---

## Quick start on a GCP VM

```bash
# 1. Clone the repository
git clone <repo-url> && cd client-selection-hfl

# 2. Run the one-shot environment setup (creates .venv, installs all deps)
bash gcp_setup.sh

# 3. Set your HuggingFace token if the dataset repo is private
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# 4. Run all four network sizes (N = 14, 35, 70, 140) – 30 rounds each
bash run_all_simulations.sh
```

Results CSVs are written to `./results/` and comparison plots to `./plots/`.

---

## Running a single experiment

```bash
source .venv/bin/activate

# Stream data, 70 IoT clients, 30 rounds, 5 % subsample (≈6 400 buildings)
python run_simulation.py --N 70 --rounds 30 --subsample 0.05

# Use a locally-downloaded CSV instead of streaming
python run_simulation.py \
    --csv_path ./data/Final_Dataset/training_dataset_with_city.csv \
    --N 70 --rounds 30 --subsample 0.05
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--csv_path` | *None* | Local CSV path. Omit to stream from HuggingFace. |
| `--data_dir` | `./data` | Root for local image chips / tile cache. |
| `--subsample` | `0.05` | Fraction of dataset rows to use (0 < x ≤ 1). |
| `--hf_token` | *env* | HuggingFace token (falls back to `$HF_TOKEN`). |
| `--N` | `70` | Number of IoT clients. |
| `--U` | `3` | Number of UAV aggregators. |
| `--rounds` | `30` | Global communication rounds. |
| `--seed` | `42` | RNG seed. |
| `--output_dir` | `./results` | CSV output directory. |
| `--plot_dir` | `./plots` | Plot output directory. |

---

## Architecture overview

```
Central Server (Tier-1)
    │  Client selection + global FedAvg (reputation-weighted)
    ▼
UAV Aggregators (Tier-2)   ×U
    │  Edge FedAvg  →  inherited reputation
    ▼
IoT Clients (Tier-3)       ×N
    Local training on seismic + aerial-image data
```

### Selection algorithm (proposed)

1. **Eligibility gate** – battery ≥ 0.2, SNR ≥ 3 dB, predicted latency ≤ T_max.
2. **Priority score** – weighted sum of battery, latency term, and a
   blended utility–reputation score (β decays from 1 → 0 over 20 rounds).
3. **UCB exploration** – `P_n + √2 · √(ln(t+1) / (N_n+1))` prevents starvation.
4. **Greedy UAV assignment** – assign by descending UCB score to the
   least-loaded feasible UAV within communication range.

### Reputation sub-scores

| Sub-score | Captures |
|-----------|---------|
| R_contrib  | Cosine similarity with the client's own EMA update history |
| R_anomaly  | Mahalanobis distance from the round's projected update mean |
| R_temp     | Success rate × inverse latency variance over last 10 rounds |

---

## Running sanity tests (no HuggingFace token needed)

```bash
python test_sanity.py
```

All I/O is mocked; tests run fully offline in a few seconds.

