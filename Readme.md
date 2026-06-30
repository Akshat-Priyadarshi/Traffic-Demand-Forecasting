# Spatial-Temporal Traffic Forecasting

> **Ensemble ML Architecture for Urban Demand Prediction** > Gradient Boosting + Autoregressive Lags & Leak-Proof Target Encoding

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

Most solutions treat this as a basic categorical regression problem. It is not. This is a **spatial-temporally aware demand forecasting problem**. Traffic demand is driven by physical geography, the cyclical nature of time, and autoregressive momentum (what happened yesterday at this exact time).

The pipeline:
- Decodes `geohash` strings into precise physical coordinate floats.
- Groups physical coordinates into spatial neighborhoods using K-Means Clustering.
- Shifts historical data to create 24-hour and 7-day autoregressive lag features.
- Applies strict Out-of-Fold (OOF) Target Encoding to establish historical demand baselines without target leakage.
- Trains a highly stable, 50/50 blended ensemble of LightGBM and CatBoost under 5-Fold Cross-Validation.

Four core commitments:
1. Never treat geohashes as opaque strings — they represent physical proximity.
2. Time is circular; it must be mapped continuously using trigonometry.
3. Target leakage is fatal — all historical baselines must be encoded strictly Out-of-Fold.
4. Every prediction maps 1:1 to the competition's expected output schema before submission.

---

## 2. System Architecture & Methodology

```text
Output        submission.csv  ←  post-processed, schema-validated
  ↑
Ensemble      Hyper-Stable Blend: LightGBM (0.5) + CatBoost (0.5)
  ↑
Training      5-Fold Stratified Cross-Validation → OOF predictions
  ↑
Features      Lags + K-Means Zones + OOF Target Encoding + Cyclical Time
  ↑
Ingestion     train.csv + test.csv → merged frame (Strictly Legal)
```

**Five engineering pillars:**

1. **Spatial Decoding** — `pygeohash` decodes strings into exact lat/lon coordinates. Sine and cosine transformations map 24-hour and 60-minute cycles onto a unit circle, ensuring the model understands that 23:55 and 00:05 are minutes apart, not hours.

2. **K-Means Spatial Zones** — Tree models struggle with diagonal boundaries. By running K-Means clustering on the coordinates, we group the map into 50 distinct "Traffic Zones," giving the trees a highly efficient geographic shortcut.

3. **Autoregressive Momentum (Lags)** — Traffic has memory. We engineered 24-hour and 7-day lag features, allowing the model to look at the exact demand for a specific location at this exact time yesterday and last week.

4. **Leak-Proof Target Encoding** — Instead of forcing the models to guess traffic baselines, we use Scikit-Learn's `TargetEncoder` inside the K-Fold loop. This gives the model explicit historical demand averages for every geohash without leaking the validation answers.

5. **The Ensemble Engine** — LightGBM excels on optimizing the continuous spatial floats and aggregation metrics; CatBoost natively handles high-cardinality categoricals and gracefully manages missing values. A uniform 50/50 blend effectively cancels out individual algorithm variance.

---

## 3. Technology Stack

| Component | Technology | Reason |
|---|---|---|
| Primary Model | LightGBM | Fast on floats, handles massive tabular arrays efficiently |
| Secondary Model | CatBoost | Native missing-value support, highly stable on categoricals |
| Clustering & CV | scikit-learn | K-Means spatial zones and leak-proof OOF Target Encoding |
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
- Load the data and engineer the 24h/7d autoregressive lags.
- Apply K-Means spatial zones and cyclical time features.
- Execute strict Out-of-Fold Target Encoding to capture historical baselines.
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
========================= 1 passed in 8.15s =========================
```

Do not upload to the competition platform until all four tests are green.

---

## 7. Design Philosophy

### Traffic Has Momentum

A model cannot accurately predict a sudden 3:00 PM surge if it doesn't know what happened at 3:00 PM yesterday. Shifting the data to create explicit lag features allows the tree models to adjust their baseline predictions dynamically based on recent momentum.

### Target Leakage is the Enemy

It is easy to achieve a false 99% accuracy on a validation set by improperly aggregating global target variables. By aggressively restricting our Target Encoders to operate exclusively inside the K-Fold split boundaries, we guarantee that our validation scores reflect reality and will generalize perfectly to unseen data.

### Validate Before You Submit

The competition platform silently rejects malformed submissions. The `pytest` suite encodes the exact schema constraints as assertions so a bad output never reaches the leaderboard.

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
The notebook did not finish executing. Re-run `solution.ipynb` completely, ensure `submission.csv` is updated in the root directory, and re-run `pytest tests/`.

---

## 9. Team & Division of Work

| Member | Responsibility |
|---|---|
| **Saksham Sinha** | Feature engineering (`src/features.py`), cyclical mapping |
| **Vishakha Priya** | LightGBM model config, K-Fold training loop |
| **Akshat Priyadarshi** | CatBoost model config, spatial decoding & K-Means clusteringg |
| **Rudra Pratap** | Out-of-Fold Target Encoding, Autoregressive lag logic |

---

## 10. Roadmap

### Post-competition

- Wrap the predictive model inside a Streamlit Dashboard to visualize urban traffic bottlenecks in real-time.
- Deploy a FastAPI backend paired with Docker Compose to serve live geographic traffic predictions.
- Experiment with graph-based neural networks (GNNs) using adjacency matrices of geohash neighbors for a third ensemble member.

---

> **Every prediction is validated against the competition schema before submission.**  
> Zero silent format failures — the test suite must be green before any upload.
