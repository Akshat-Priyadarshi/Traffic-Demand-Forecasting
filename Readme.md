# GridLock Challenge: Spatial-Temporal Traffic Forecasting

> **Ensemble ML Architecture for Urban Demand Prediction** > Gradient Boosting + Historical Target Aggregation & Spatial Feature Engineering

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

Most solutions treat this as a basic categorical regression problem. It is not. This is a **spatial-temporally aware demand forecasting problem** where the physical geography of each cell and the historical traffic patterns of that exact location are the absolute strongest predictors of future demand.

The pipeline:
- Decodes `geohash` strings into precise physical coordinate floats.
- Engineers explicit **Historical Target Aggregations** to give the models a baseline truth.
- Trains a highly stable, 50/50 blended ensemble of LightGBM and CatBoost under 5-Fold Cross-Validation.
- Operates **100% within hackathon rules**, relying strictly on the provided `train.csv` to prevent data leakage and disqualification.

Four core commitments:
1. Never treat geohashes as opaque strings — they represent physical proximity.
2. Trees prefer discrete historical facts over continuous circular mathematics.
3. Every prediction maps 1:1 to the competition's expected output schema before submission.
4. Reproducible results — same seed, same folds, same output on every run.

---

## 2. System Architecture & Methodology

```text
Output        submission.csv  ←  post-processed, schema-validated
  ↑
Ensemble      Hyper-Stable Blend: LightGBM (0.5) + CatBoost (0.5)
  ↑
Training      5-Fold Stratified Cross-Validation → OOF predictions
  ↑
Features      Spatial decoding + Historical Target Aggregations
  ↑
Ingestion     train.csv + test.csv → merged frame (Strictly Legal)
```

**Four engineering pillars:**

1. **Spatial Decoding** — `pygeohash` decodes each geohash string into exact latitude/longitude coordinates, giving tree models meaningful continuous spatial distances rather than isolated categories.

2. **Historical Target Aggregations (The Keystone)** — Instead of forcing the models to "guess" traffic demand based purely on weather or road types, we explicitly mapped the historical traffic baselines `(geo_mean, geo_median, geo_std, and hour_mean)` directly from the training data onto the test set. This provides a massive, highly stable predictive signal.

3. **The Ensemble Engine** — LightGBM excels on optimizing the continuous spatial floats and aggregation metrics; CatBoost natively handles high-cardinality categoricals and gracefully manages missing values. A blended prediction effectively cancels out individual model errors.

4. **Strict Rule Compliance & Generalization** — Automated weighting optimizers and external data leaks were strictly avoided. A robust 50/50 uniform blend ensures the model generalizes perfectly to the unseen leaderboard data without overfitting the validation set.

---

## 3. Technology Stack

| Component | Technology | Reason |
|---|---|---|
| Primary Model | LightGBM | Fast on floats, handles massive tabular arrays efficiently |
| Secondary Model | CatBoost | Native missing-value support, highly stable on categoricals |
| Spatial Decoding | pygeohash | Converts geohash → lat/lon for continuous spatial features |
| Validation | pytest | Automated output schema checks before upload |
| Execution | Jupyter Notebook | Reproducible, step-by-step training with visible K-Fold scores |
| Environment | Python 3.10 + venv | Isolated, conflict-free dependency management |

---

## 4. Project Structure

```
traffic_prediction_challenge/
│
├── solution.ipynb           ← Main execution: training, aggregation mapping, blending
├── Approach.txt             ← Executive summary of methodology (For Judges)
├── requirements.txt         ← All project dependencies
├── README.md                ← This file
│
├── src/
│   ├── __init__.py
│   └── features.py          ← Spatial feature engineering module
│
├── tests/
│   ├── __init__.py
│   └── test_submission.py   ← Automated formatting + schema validation suite
│
└── data/                    ← ⚠ Git-ignored — provision manually (see Step 4 below)
    ├── train.csv
    └── test.csv
```

---

## 5. Setup & Installation

Follow these steps in order. Use a Linux or WSL terminal throughout.

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/Akshat-Priyadarshi/GridLock-Challenge2.0.git
cd GridLock-Challenge2.0
```

**Step 2 — Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3 — Install dependencies:**
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

---

## 6. Execution & Testing

**Step 1 — Launch Jupyter and run the notebook:**
```bash
jupyter notebook solution.ipynb
```
Execute all cells in order. The notebook will:
- Load and merge the hackathon data.
- Calculate and map historical target aggregations based strictly on the training split.
- Engineer spatial coordinate features.
- Train LightGBM and CatBoost under 5-Fold Cross-Validation
- Blend predictions and export `submission.csv`

**Step 2 — Validate the output before uploading:**
```bash
pytest tests/
```
The test suite checks:
- Output dimensions are exactly `41778 × 2`
- Column headers exactly match the required schema
- No null values anywhere in the submission

A fully passing run looks like:
```
========================= 1 passed in 3.73s =========================
```

Do not upload to the competition platform until all four tests are green.

---

## 7. Design Philosophy

### Spatial Context Is Not Optional

A geohash string like `qp09` is meaningless to a gradient boosting tree. Decoded into `(1.312°N, 103.848°E)`, it becomes a real point in space. The model can then learn that demand at one cell correlates with demand at its physical neighbours — a relationship that raw hash strings destroy.

### History Repeats Itself

Gradient boosting models operate by making discrete splits. While complex cyclical trigonometry (sine/cosine) sounds advanced, trees actually struggle to optimize circular mathematics. Giving the tree explicit historical baselines `(e.g., "This specific geohash averages 0.04 demand at 2:00 PM")` gives the algorithm an immediate, highly accurate foundation to adjust from based on weather or road types.

### Validate Before You Submit

The competition platform silently rejects malformed submissions. The pytest suite encodes the exact schema constraints as assertions so a bad output never reaches the leaderboard.

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'pygeohash'`
Your virtual environment is not activated. Run:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `FileNotFoundError: data/train.csv`
The `data/` folder is git-ignored and must be provisioned manually. See Step 4 of Setup.

### Jupyter kernel dies during CatBoost training
CatBoost allocates significant RAM during tree construction. Close other applications, or reduce iterations in the CatBoost config from 2500 to 1500 for a lower-memory run.

### `pytest` reports dimension mismatch (not 41778 rows)
The notebook did not finish executing. Re-run solution.ipynb completely, ensure submission.csv is updated in the root directory, and re-run pytest tests/.

---

## 9. Team & Division of Work

| Member | Responsibility |
|---|---|
| **Saksham Sinha** | Feature engineering (`src/features.py`), geohash spatial decoding |
| **Vishakha Priya** | LightGBM model config, K-Fold training loop |
| **Akshat Priyadarshi** | CatBoost model config, ensemble blending, output post-processing |
| **Rudra Pratap** | Test suite (`tests/test_submission.py`), historical target aggregation mapping |

---

## 10. Roadmap

### Post-competition

- Wrap the predictive model inside a Streamlit Dashboard to visualize urban traffic bottlenecks in real-time.
- Deploy a FastAPI backend paired with Docker Compose to serve live geographic traffic predictions.
- Experiment with graph-based spatial features: adjacency matrix of geohash neighbors.
- Evaluate TabNet as a third ensemble member for attention-based feature selection.

---

> **Every prediction is validated against the competition schema before submission.**  
> Zero silent format failures — the test suite must be green before any upload.