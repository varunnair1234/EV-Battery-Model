# 🔋 EV Battery Model

A machine learning project to **build, train, and evaluate models that predict EV battery health and degradation** from real-world driving + charging data.  
Primary focus: **State of Health (SOH) regression** (and extensible to range degradation and service-risk classification).

---

## ✨ What This Repo Does

- Ingests battery/vehicle logs (CSV) into a clean processed dataset
- Builds features that correlate with degradation (cycles, throughput, fast-charge ratio, temperature exposure, etc.)
- Trains:
  - **Baselines (scikit-learn)** for a strong floor
  - **PyTorch MLP** as a first neural model
  - (Optional) sequence models later (GRU/Transformer) for session history
- Evaluates using MAE/RMSE/R² and writes reproducible reports
- Saves trained models and artifacts to versioned folders

---

## 📌 Targets (Choose One to Start)

Common prediction targets:
- **SOH (%)** (recommended starting point)
- **Capacity remaining (kWh)**
- **Estimated range (miles)**
- **Service risk / anomaly detection** (classification)

---

## 🧱 Repo Structure
ev-battery-model/
README.md
requirements.txt
.gitignore

data/
raw/                  # original CSVs (never edited)
processed/            # generated clean datasets
external/             # optional public datasets

notebooks/
01_eda.ipynb
02_baseline_model.ipynb
03_pytorch_model.ipynb

src/
config.py

