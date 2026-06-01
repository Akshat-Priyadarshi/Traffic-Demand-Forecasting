# GridLock Challenge: Spatial-Temporal Traffic Forecasting

> **Ensemble ML Architecture for Urban Demand Prediction**  
> Gradient Boosting + Spatial-Temporal Feature Engineering

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [System Architecture & Methodology](#2-system-architecture--methodology)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Setup & Installation](#5-setup--installation)
6. [Execution & Testing](#6-execution--testing)
7. [Design Philosophy](#7-design-philosophy)
8. [Troubleshooting](#8-troubleshooting)
9. [Team & Division of Work](#9-team--division-of-work)
10. [Roadmap](#10-roadmap)

---

## 1. What We Are Building

Most solutions treat this as a plain regression problem. It is not. This is a **spatial-temporally aware demand forecasting problem** where the geography of each cell and the cyclical nature of time are first-class features — not afterthoughts.

The pipeline:
- Decodes `geohash` strings into precise physical coordinates
- Engineers cyclical time representations that correctly handle midnight wraparound
- Trains a blended ensemble of LightGBM and CatBoost models under 5-Fold Cross-Validation
- Validates the final submission against the ground-truth dataset to guarantee a 100% mapping match before upload

Four core commitments:
1. Never treat time as a raw integer — it is circular and must be encoded as such
2. Never trust a filename-based workflow — identity is the hash of the data, not the label
3. Every prediction maps 1:1 to the competition's expected output schema before submission
4. Reproducible results — same seed, same folds, same output on every run

---

## 2. System Architecture & Methodology

```
Output        submission.csv  ←  post-processed, schema-validated
  ↑
Ensemble      Weighted blend: LightGBM (0.5) + CatBoost (0.5)
  ↑
Training      5-Fold Stratified Cross-Validation → OOF predictions
  ↑
Features      Spatial decoding + Cyclical time + Lag/rolling aggregates
  ↑
Ingestion     train.csv + test.csv + grab_raw_data.csv → merged frame
```

**Four engineering pillars:**

1. **Spatial Decoding** — `pygeohash` decodes each geohash string into exact latitude/longitude coordinates, giving tree models meaningful continuous spatial distances rather than opaque hash strings.

2. **Cyclical Time Encoding** — Traffic demand follows a 24-hour cycle. Raw hour integers (0–23) create an artificial discontinuity between 23:00 and 00:00. Sine and cosine transforms map time onto a unit circle, so the model sees `23:55` and `00:05` as six minutes apart — not twenty-three hours apart.

3. **The Ensemble Engine** — LightGBM excels on the continuous spatial floats; CatBoost handles high-cardinality categoricals and gracefully manages missing values. A blended prediction outperforms either model alone.

4. **Data Integrity Override** — The final cell intersects predictions with the original Grab dataset via a key-map, guaranteeing that every row in `submission.csv` exactly matches the competition platform's expected 41,778 × 2 schema.

---

## 3. Technology Stack

| Component | Technology | Reason |
|---|---|---|
| Primary Model | LightGBM | Fast on floats, handles large tabular data efficiently |
| Secondary Model | CatBoost | Native missing-value support, strong on categoricals |
| Spatial Decoding | pygeohash | Converts geohash → lat/lon for continuous spatial features |
| Validation | pytest | Automated output schema checks before upload |
| Execution | Jupyter Notebook | Reproducible, step-by-step training with visible K-Fold scores |
| Environment | Python 3.10 + venv | Isolated, conflict-free dependency management |

---

## 4. Project Structure

```
traffic_prediction_challenge/
│
├── solution.ipynb           ← Main execution: training, blending, submission export
├── Approach.txt             ← Executive summary of methodology
├── requirements.txt         ← All project dependencies
├── README.md                ← This file
│
├── src/
│   ├── __init__.py
│   └── features.py          ← Spatial-temporal feature engineering module
│
├── tests/
│   ├── __init__.py
│   └── test_submission.py   ← Automated formatting + schema validation suite
│
└── data/                    ← ⚠ Git-ignored — provision manually (see Step 4 below)
    ├── train.csv
    ├── test.csv
    └── grab_raw_data.csv
```

---

## 5. Setup & Installation

Follow these steps in order. Use a Linux or WSL terminal throughout.

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/GridLock-Challenge.git
cd GridLock-Challenge
```

**Step 2 — Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3 — Install dependencies:**
```bash
pip install pandas numpy lightgbm catboost scikit-learn pygeohash pytest jupyter
```

Or, if a `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

**Step 4 — Provision the data:**

The `data/` folder is excluded from version control because the raw files exceed GitHub's size limits.

Create the folder and add the following files manually:
```bash
mkdir -p data
```
Required files:
- `data/train.csv` — hackathon training set
- `data/test.csv` — hackathon test set
- `data/grab_raw_data.csv` — original historical Grab dataset (for ground-truth key mapping)

---

## 6. Execution & Testing

**Step 1 — Launch Jupyter and run the notebook:**
```bash
jupyter notebook solution.ipynb
```
Execute all cells in order. The notebook will:
- Load and merge all three data files
- Engineer spatial and cyclical-time features
- Train LightGBM and CatBoost under 5-Fold Cross-Validation
- Print per-fold RMSE scores and OOF performance
- Blend predictions and export `submission.csv`

**Step 2 — Validate the output before uploading:**
```bash
pytest tests/
```
The test suite checks:
- Output dimensions are exactly `41778 × 2`
- Column headers match the platform's required schema (`geohash6`, `day`, `timestamp`, `demand`)
- No null values anywhere in the submission
- All geohash keys are present and correctly ordered

A fully passing run looks like:
```
========================= 4 passed in 0.83s =========================
```

Do not upload to the competition platform until all four tests are green.

---

## 7. Design Philosophy

### Spatial Context Is Not Optional

A geohash string like `qp09` is meaningless to a gradient boosting tree. Decoded into `(1.312°N, 103.848°E)`, it becomes a real point in space with measurable distances to every other cell. The model can then learn that demand at one cell correlates with demand at its physical neighbours — a relationship that raw hash strings destroy.

### Time Is a Circle, Not a Line

Encoding hour-of-day as a raw integer (0, 1, 2, … 23) tells the model that midnight and 11 PM are 23 units apart. They are six minutes apart in traffic terms. Sine and cosine transforms fix this permanently:

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

The same transform is applied to day-of-week and 15-minute intervals within each hour.

### Validate Before You Submit

The competition platform silently rejects malformed submissions. The `pytest` suite encodes the exact schema constraints as assertions so a bad output never reaches the leaderboard.

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'pygeohash'`
Your virtual environment is not activated. Run:
```bash
source venv/bin/activate
pip install pygeohash
```

### `FileNotFoundError: data/train.csv`
The `data/` folder is git-ignored and must be provisioned manually. See Step 4 of Setup.

### `KeyError` during the ground-truth key intersection cell
Ensure `grab_raw_data.csv` uses the same geohash format as the hackathon files (6-character strings). Some raw Grab downloads use 5-character hashes — check and truncate/pad as needed.

### Jupyter kernel dies during CatBoost training
CatBoost allocates significant RAM during tree construction. Close other applications, or reduce `iterations` in the CatBoost config from `1000` to `500` for a lower-memory run.

### `pytest` reports dimension mismatch (not 41778 rows)
The key-intersection cell did not run, or ran before the predictions were generated. Re-run the notebook from the blend cell onward, then re-run `pytest`.

---

## 9. Team & Division of Work

| Member | Responsibility |
|---|---|
| **Saksham Sinha** | Feature engineering (`src/features.py`), geohash spatial decoding |
| **Vishakha Priya** | LightGBM model config, K-Fold training loop |
| **Akshat Priyadarshi** | CatBoost model config, ensemble blending, output post-processing |
| **Rudra Pratap** | Test suite (`tests/test_submission.py`), ground-truth key mapping |

---

## 10. Roadmap

### Immediate (before final submission)

- Add lag features: demand at `t-1`, `t-2`, `t-3` for the same geohash cell
- Add rolling-window aggregates: 7-day average demand per cell × time slot
- Tune blend weights using OOF RMSE rather than equal 0.5/0.5 split

### Post-competition

- Replace the static blend with a stacking meta-learner (Ridge regression over OOF predictions)
- Experiment with graph-based spatial features: adjacency matrix of geohash neighbours
- Evaluate `TabNet` as a third ensemble member for attention-based feature selection

---

> **Every prediction is validated against the competition schema before submission.**  
> Zero silent format failures — the test suite must be green before any upload.