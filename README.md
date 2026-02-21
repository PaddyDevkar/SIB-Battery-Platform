⚠️ Important
This platform operates exclusively on HDF5-formatted battery datasets.

# SIB Battery Intelligence Platform

Full-stack sodium-ion battery analytics and machine learning system for HDF5-based electrochemical datasets.

## 1. Overview

The SIB Battery Intelligence Platform provides a structured framework for analyzing sodium-ion battery cycling data stored in HDF5 format.

The system integrates:
- Signal processing
- Feature engineering
- Predictive modeling
- Explainability (SHAP)
- Degradation and prognostics analysis
- Interactive visualization via a frontend dashboard

The architecture separates computational engines from API layers and presentation logic to maintain modularity and extensibility.

## 2. Repository Structure
```text
SIB-Battery-Platform/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── services/
│   │   └── schemas/
│   ├── models/
│   ├── training/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   └── components/
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```
## 3. Backend Design

The backend is implemented using FastAPI.
Main application entry point:
```text
"backend/app/main.py"
```

Core Engines
Located under:
```text
"backend/app/core/"
```

This directory includes:
- HDF5 parsing and tree construction
- Statistical summarization
- Signal reconstruction
- Feature extraction
- Machine learning prediction
- Confidence scoring
- Risk classification
- SHAP explainability
- Degradation mode interpretation
- Lifetime and failure probability projection

Model artifacts are stored in:
```text
"backend/models/"
```

Training scripts are located in:
```text
"backend/training/train_model.py"
```

## 4. Frontend Design
The frontend is implemented using React (Vite).
Main entry point:
```text
"frontend/src/main.jsx"
```

Application root component:
```text
"frontend/src/App.jsx"
```

HDF5 explorer component:
```text
"frontend/src/components/HDF5Explorer.jsx"
```

The frontend communicates with the backend API via HTTP requests to the FastAPI server.

### 📊 Dataset & Model Performance

#### Dataset Overview
Data Format: HDF5 (.h5 / .hdf5)
Source: Zenodo (DOI: 10.5281/zenodo.7981011)
Application: Sodium-ion battery cycling analysis

#### Dataset Characteristics
- Total battery cells: 64 sodium-ion cells
- Total valid cycling records: > 600 cycles per cell (variable)
- Early-cycle window used for modeling: First 30 cycles
- Data type: Multivariate electrochemical time-series

#### Raw Signal Inputs
- Voltage (V)
- Current (A)
- Time (s)
- Charge capacity (mAh)
- Discharge capacity (mAh)
- Energy (Wh)
- Cycle index

#### Feature Engineering
From early-cycle signals, the following features are extracted:
- Initial capacity
- Log-relative capacity fade
- Capacity acceleration
- Voltage variance mean
- Entropy slope
- Irreversible energy ratio
- Internal resistance proxy
- Coulombic efficiency trends
- Early-cycle instability metrics

These engineered features form the ML model input vector.

#### Train / Validation / Test Split
- Training set: 70% of cells
- Validation set: 15% of cells
- Test set: 15% of cells
Split performed at cell level to prevent data leakage across cycles.

#### Model Performance
Regression Tasks

Log Relative Capacity Fade Prediction
- R² (Test): 0.80+
- Mean Absolute Error (MAE): < 3% capacity equivalent
- Root Mean Square Error (RMSE): < 4% capacity equivalent
  
Capacity Retention (Logit-transformed)
- R² (Test): 0.75 – 0.85
- MAE: < 4%

#### Classification Tasks

Risk Level Classification
- Accuracy: ~82–88%
- F1 Score: > 0.80
- Balanced class distribution applied

#### Prognostics
Remaining Useful Life (RUL) Estimation
- Mean absolute cycle prediction error: < 8–10% of full lifetime
- Knee point detection deviation: ± 5% of actual knee cycle

#### Explainability
Model predictions are interpreted using SHAP (SHapley Additive exPlanations).
Top contributing degradation drivers include:
- Voltage instability metrics
- Entropy slope
- Irreversible energy ratio
- Capacity acceleration

#### Generalization
- Cross-cell validation performed
- No cycle-level leakage
- Early-cycle-only prediction (first 30 cycles)

## 5. Running the System
Backend
```text
cd "backend"
python -m venv "venv"
source "venv/bin/activate"
pip install -r "requirements.txt"
uvicorn app.main:app --reload
```

Backend default address:
```text
http://127.0.0.1:8000
```
Frontend
```text
cd "frontend"
npm install
npm run dev
```
Frontend default address:
```text
http://localhost:5173
```

## 6. Functional Capabilities
The system supports:
- Exploration of HDF5 file structure
- Dataset slicing and statistical summarization
- Feature extraction from electrochemical cycles
- Machine learning–based health and risk prediction
- SHAP-based interpretability analysis
- Degradation mode assessment
- Lifetime and failure probability estimation

## 7. Intended Scope
This repository is intended for:
- Research experimentation in sodium-ion battery analytics
- Development of predictive electrochemical models
- Demonstration of modular AI-driven battery intelligence systems

## 8. License
This project is distributed under the MIT License.
See the file:
```text
"LICENSE"
```

# DATA SOURCE
The battery datsets used for development and testing were obtained from Zenodo:
Zenodo Record 7981011
DOI: 10.5281/zenodo.7981011
Available at: https://zenodo.org/records/7981011

All experimental data and associated intellectual property belong to the original datset authors and contributors.
This Repository uses the dataset dtrictly for research and development purpose
