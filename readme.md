#  Churn ML Pipeline — Prod-Style Layout

**End-to-end modular pipeline for churn prediction** (10k+ records) built with reproducible steps, checkpoints, and monitoring hooks.

This project demonstrates a production-style machine learning workflow:  
data validation → feature engineering → model training → evaluation → artifact tracking.

---

## Project Overview

**Goal:** Predict booking cancellations (customer churn) for **ABC Hotels** using a neural network pipeline with full preprocessing, version snapshots, and drift-readiness.

| Stage | Description |
|--------|--------------|
| Data Ingestion | Load and validate raw dataset (`project_data.csv`) |
| Preprocessing | Imputation, scaling, encoding via `ColumnTransformer` |
| Modeling | Two dense MLP classifiers compared (`(64,32)` vs `(32,16)`) |
| Evaluation | ROC-AUC, learning curves, calibration, permutation importance |
| Artifacts | Reproducible metrics, plots, and reports saved under `/artifacts` |

---

##  Stack

| Layer | Tools |
|--------|--------|
| Language | Python 3.x |
| Core ML | `scikit-learn`, `NumPy`, `pandas` |
| Visualization | `matplotlib` |
| Pipeline Design | `Pipeline`, `ColumnTransformer` |
| Reproducibility | `random_state`, saved metrics & artifacts |

---

##  Highlights

-  **Reproducible runs** with fixed seeds and modular configs  
-  **Schema & version snapshots** for consistent preprocessing  
-  **Drift-check ready**: easily extend to monitor input schema or distribution drift  
-  **Learning & calibration curves** for generalization insight  
-  **Permutation importance** for explainability  
-  **Artifact persistence**: metrics, plots, and reports automatically saved

---

##  Key Results

| Metric | Model A (64,32) | Model B (32,16) |
|---------|-----------------|-----------------|
| ROC-AUC (valid) | 0.921 | 0.912 |
| Accuracy | 0.861 | 0.854 |
| F1 Score | 0.771 | 0.764 |

**Final Model:** Model A (64, 32) — best AUC and calibration balance.  

Generated figures:
- `roc_curve_A_vs_B.png`
- `calibration_curve_A_vs_B.png`
- `learning_curve_final_model.png`
- `confusion_matrix_final.png`
- `permutation_importance_top20.png`

---

##  Running the Pipeline

### 1️⃣ Clone the repo 

git clone <TODO: add repo link> 
cd churn-ml-pipeline 
  
### 2️⃣ Install dependencies 
pip install -r requirements.txt 
 
### 3️⃣ Run the pipeline 
python part3.ipynb   # or open in JupyterLab / VS Code 
  
### 4️⃣ View artifacts 
All outputs (metrics, figures, reports) are stored under: 



## Enhancement Idea 

-Data drift monitor  
-Integrate with Evidently AI or custom drift statistics   
-Experiment tracking  
-Add MLflow or Weights & Biases  
-Schema enforcement  
-Add Pydantic or Great Expectations  
-Model registry  
-Store final models & metrics snapshots for versioning deployment  
-Export pipeline via joblib → deploy as REST API or batch job  
 
 
├── part3.ipynb              # Main reproducible notebook  
├── project_data.csv      # Dataset (10k+ records)  
├── artifacts/                  # Saved metrics, figures, reports  
├── requirements.txt      # Dependencies  
├── final_project.pdf      # Condensed report  
└── README.md           # This file  

## Author   
Richard Wilders   
Merrimack College — Machine Learning & AI Project   
richard16meyer@gmail.com 

