# AI-Powered Water Intelligence & Disaster Resilience Platform
## Full Deployment Guide & Research System Documentation

This repository provides an enterprise-ready, research-grade deployment wrapping the existing, audited machine learning models and hydrological reports for the **IndoFloods** benchmark.

---

### Architectural Overview

```
                      +------------------------------------------+
                      |      React + Vite + Tailwind CSS         |
                      |   Hydrological Intelligence Dashboard    |
                      |        http://localhost:5173             |
                      +--------------------+---------------------+
                                           |  REST API (JSON)
                                           v
                      +--------------------+---------------------+
                      |         FastAPI Python Backend           |
                      |   Pydantic V2 + Validation + Routing     |
                      |   http://127.0.0.1:8000 (Swagger: /docs) |
                      +----+-------------------+---------------+--+
                           |                   |               |
             +-------------v----+     +--------v-------+  +----v--------------------+
             |  PostgreSQL / DB |     | IndoFloods Data|  |  Existing ML Pipelines   |
             | SQLAlchemy Models|     | 4,548 Events   |  | .joblib Scikit-Learn    |
             | Prediction Logs  |     | 155 Gauges     |  | 113 Controlled Features |
             +------------------+     +----------------+  +-------------------------+
```

- **Frontend**: React 19 + Vite + Tailwind CSS + Recharts + Lucide Icons
- **Backend**: FastAPI + Uvicorn + Pydantic + SQLAlchemy
- **Database**: PostgreSQL (with automated fallback to local SQLite if PostgreSQL credentials are not yet initialized)
- **ML Artifacts**: Pre-trained Scikit-Learn Pipelines (`models/regression_controlled/` and `models/classification/`)
- **Scientific Integrity Status**: 27 PASS, 0 FAIL (`reports/integrity_audit/final_integrity_audit.csv`)

---

### Preserved Research Files (Read-Only Compliance)

All pre-existing datasets, training artifacts, and evaluation reports have been strictly preserved without retraining, renaming, or modification:

- `data/processed/`: `indofloods_integrated.csv`, `indofloods_modeling_base.csv`, target regression and classification datasets
- `data/splits/`: Grouped split sets (`controlled_*_X_train.csv`, `controlled_*_X_test.csv`, `groups_train.csv`, etc.)
- `models/regression_controlled/`:
  - `controlled_peak_flood_level_gradient_boosting.joblib`
  - `controlled_peak_discharge_gradient_boosting.joblib`
  - `controlled_flood_volume_gradient_boosting.joblib`
- `models/classification/`:
  - `gradient_boosting.joblib`
- `reports/`:
  - SHAP feature attribution (`reports/shap/`)
  - Residual distributions & summary (`reports/residuals/`)
  - Error analysis & regime diagnostics (`reports/error_analysis/`, `reports/gauge_diagnostics/`)
  - Audit logs (`reports/integrity_audit/final_integrity_audit.csv`)
- `src/`: Existing model training, profiling, and audit scripts

---

### Prerequisites

- **Python**: 3.10+ (Python 3.14 / 3.13 confirmed with FastAPI, SQLAlchemy, Scikit-Learn, Pandas)
- **Node.js**: v18+ (Node v24.15 and npm 11.12 confirmed)
- **PostgreSQL**: Local or remote instance (PostgreSQL service confirmed running on port 5432)

---

### Step-by-Step Windows PowerShell Commands

#### 1. Configure PostgreSQL Database

Open PowerShell and connect to your PostgreSQL instance to create the database:

```powershell
# If using psql CLI:
psql -U postgres -h localhost -p 5432 -c "CREATE DATABASE water_intelligence;"

# Or using pgAdmin:
# Create a new database named: water_intelligence
```

Create your `.env` configuration file in the project root:

```powershell
# Copy the example environment file
Copy-Item .env.example .env

# Edit .env with your PostgreSQL credentials:
# DATABASE_URL=postgresql://postgres:<your_password>@localhost:5432/water_intelligence
```

> **Note**: If PostgreSQL is not configured immediately, the platform automatically utilizes a local SQLite database (`predictions.db`) to allow immediate full testing without interruption.

---

#### 2. Install Backend Dependencies & Verify Models

```powershell
# Navigate to project root
cd "C:\Users\kvaad\OneDrive\Desktop\AI water-intelligence"

# Install backend dependencies (already satisfied in existing python environment)
py -3.14 -m pip install -r backend/requirements.txt

# Run deployment model check to verify all 113 features load correctly:
py -3.14 src/deployment_model_check.py
```

---

#### 3. Run Backend API Server

```powershell
# Start FastAPI backend with Uvicorn (PowerShell)
$env:PYTHONPATH = "."
py -3.14 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend will start and log:
```
[INFO] water_intelligence.api: Initializing Water Intelligence API and Database...
[INFO] water_intelligence.database: Database schema initialized successfully.
[INFO] Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

- **Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

#### 4. Install Frontend Dependencies & Start React App

In a separate PowerShell window:

```powershell
# Navigate to frontend directory
cd "C:\Users\kvaad\OneDrive\Desktop\AI water-intelligence\frontend"

# Install npm packages
npm install

# Start Vite development server
npm run dev
```

The frontend will start at:
- **Application URL**: [http://localhost:5173](http://localhost:5173)

---

#### 5. Verify the API End-to-End

In PowerShell, you can execute a test verification script or run `curl`:

```powershell
# Test Health Endpoint
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get

# Test Retrospective Historical Replay (Peak Flood Level)
$body = @{ event_id = "INDOFLOODS-gauge-1010-1" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/flood-level" -Method Post -Body $body -ContentType "application/json"

# Test Multi-Target Prediction (Level + Discharge + Volume + Type)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/all" -Method Post -Body $body -ContentType "application/json"

# Test SHAP Feature Attribution
$explainBody = @{ target = "peak_flood_level"; top_n = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/explain" -Method Post -Body $explainBody -ContentType "application/json"
```

---

### Core API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model loading status, DB connectivity, dataset counts |
| `GET` | `/metadata/features` | 113 grouped features with units, types, and defaults |
| `GET` | `/analytics/summary` | Overall benchmark metrics (MAE, RMSE, R², 27 PASS audit) |
| `GET` | `/analytics/shap/{target}` | SHAP feature attributions and directional contributions |
| `GET` | `/analytics/residuals/{target}` | Parity scatter points and residual error histograms |
| `GET` | `/analytics/error-analysis/{target}`| Extreme outlier events and regime error comparisons |
| `GET` | `/events` | Searchable, paginated historical events from IndoFloods |
| `GET` | `/events/{event_id}` | Complete predictor and observed target values for an event |
| `GET` | `/gauges` | List of 155 river monitoring stations with catchment specs |
| `GET` | `/gauges/{gauge_id}` | Detailed baseline characteristics for a specific gauge |
| `POST` | `/predict/flood-level` | Forecast Peak Flood Level (`m`) |
| `POST` | `/predict/discharge` | Forecast Peak Discharge Q (`cumec`) |
| `POST` | `/predict/volume` | Forecast Flood Volume (`cumec`) |
| `POST` | `/predict/flood-type` | Binary classification: `Flood` vs `Severe Flood` |
| `POST` | `/predict/all` | Simultaneous multi-target prediction |
| `POST` | `/explain` | Audited SHAP importance ranking |
| `GET` | `/history` | Fetch persistent prediction history from database |
| `DELETE`| `/history/{id}` | Delete a prediction record |

---

### Platform Pages & User Guide

1. **Dashboard (`/`)**:
   - Visualizes total flood events (4,548), gauge coverage (155), event severity breakdown (64.2% Standard vs 35.8% Severe), and audited regression metrics (Peak Flood Level R² = 0.885, MAE = 31.10 m).
2. **Flood Prediction**:
   - **Mode A: Historical Event Replay**: Search and select from 4,548 genuine events across India to retrospectively replay model predictions against observed ground truth. Labeled with scientific disclaimer.
   - **Mode B: Scenario Prediction**: Use 155 Gauge Catchment Templates to auto-populate the 103 static catchment, bioclimatic, and socioeconomic parameters, then adjust 10-day antecedent rainfall (T1d to T10d) sliders to simulate custom watershed responses.
3. **Flood Type Classification**:
   - Classifies events into Flood vs Severe Flood with confidence metrics. Documents the architectural distinction between 118 classification features and the 113 controlled regression features (threshold fields excluded to prevent leakage).
4. **Model Explainability (SHAP)**:
   - Interactive ranking of top features (Relief Ratio, Antecedent Precip, Bioclimatic temperatures, Basin Krishna). Displays directional positive vs negative contributions with explicit disclaimer: *"SHAP values indicate model contribution, not causal relationships."*
5. **Hydrological Analytics**:
   - Actual vs Predicted parity dispersion, residual error distribution histograms, and error analysis across relief regimes and outlier events.
6. **Prediction History**:
   - Real-time audit log of past predictions with timestamp, mode, input identifiers, predicted values, and actual residuals stored in PostgreSQL / SQLAlchemy.

---

### Research Integrity & Safety Rules

- **Zero-Retraining**: The system executes pre-trained pipelines directly (`model.predict(df)`).
- **Exact Features**: Exactly 106 numerical + 7 categorical features are supplied in DataFrame format to the regression pipelines. Preprocessing is encapsulated inside the joblib artifacts.
- **Scientific Disclaimers**: All retrospective replay and scenario outputs carry explicit non-operational disclaimers.
