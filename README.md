# 🔋 SIB Battery Intelligence Platform

Full-stack AI-powered Sodium-Ion Battery analytics system.

---

## 🚀 Features

### 🧠 AI Battery Analyzer
- Health Score
- SOH (Energy & Power)
- Remaining Useful Life
- Degradation Mode
- Failure Probability
- AI Interpretation

### 📁 HDF5 Explorer
- Interactive file tree
- Dataset visualization
- Zoom & Pan
- Log scale
- FFT view
- Smoothing
- Downsampling
- Statistical summary

---

## 🏗 Architecture

Frontend:
- React (Vite)
- Plotly.js

Backend:
- FastAPI
- h5py
- NumPy

---

## ⚙️ Installation

### Backend (SIB-Predictor)

cd SIB-Predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

API runs on:
http://127.0.0.1:8000

---

### Frontend (sib-dashboard)

cd sib-dashboard 
npm install
npm run dev

App runs on:
http://localhost:5173

---

## 🔌 API Endpoints

- POST /predict
- POST /hdf5/structure
- POST /hdf5/dataset

---

## 📦 Deployment

Supports:
- Docker
- Cloud deployment (Render, AWS, etc.)
