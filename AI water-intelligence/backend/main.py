import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.database import get_db, init_db, is_sqlite_fallback, active_engine
from backend.models import PredictionRecord, PredictionInput
from backend.schemas import (
    PredictionRequest,
    SinglePredictionResponse,
    ClassificationPredictionResponse,
    MultiPredictionResponse,
    ExplainRequest,
    ExplainResponse,
    PaginatedEventsResponse,
    GaugeSummary,
    AnalyticsSummaryResponse,
    PredictionHistoryItem
)
from backend.prediction_service import service

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("water_intelligence.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Water Intelligence API and Database...")
    init_db()
    logger.info("Water Intelligence Platform Backend is ready.")
    yield


app = FastAPI(
    title="AI Water Intelligence & Disaster Resilience Platform API",
    description="Research-grade hydrological intelligence platform for predictive flood modeling, explainability (SHAP), error analytics, and retrospective event replay using IndoFloods dataset.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for frontend applications (Vite standard ports: 5173, 3000, 8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 1. HEALTH & METADATA ENDPOINTS
# ==============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint indicating model loading status and database connectivity."""
    db_status = "connected (SQLite fallback)" if is_sqlite_fallback else "connected (PostgreSQL)"
    return {
        "status": "healthy",
        "database": db_status,
        "models_loaded": {
            "peak_flood_level": "peak_flood_level" in service.regression_models,
            "peak_discharge": "peak_discharge" in service.regression_models,
            "flood_volume": "flood_volume" in service.regression_models,
            "flood_type_classification": service.classification_model is not None
        },
        "regression_features_count": len(service.regression_features),
        "total_historical_events": len(service.events_by_id),
        "total_gauges": len(service.gauges_dict)
    }


@app.get("/metadata/features", tags=["System"])
def get_features_metadata():
    """Returns grouped feature definitions, types, options, and defaults for scenario building."""
    return service.get_feature_metadata()


# ==============================================================================
# 2. ANALYTICS & RESEARCH REPORT ENDPOINTS
# ==============================================================================

@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse, tags=["Analytics"])
def get_analytics_summary():
    """Returns overall research analytics: total flood events, severe flood counts, gauge coverage, integrity audit, and model metrics."""
    return service.get_analytics_summary()


@app.get("/analytics/shap/{target}", response_model=ExplainResponse, tags=["Analytics"])
def get_shap_analytics(target: str = "peak_flood_level", top_n: int = Query(15, ge=5, le=50)):
    """Fetches audited SHAP importance and directional effects for a specified target."""
    try:
        return service.get_shap_explanation(target, top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/residuals/{target}", tags=["Analytics"])
def get_residuals_analytics(target: str = "peak_flood_level"):
    """Fetches residual metrics, sample actual vs predicted points, and error distribution histogram."""
    try:
        return service.get_residuals_data(target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/error-analysis/{target}", tags=["Analytics"])
def get_error_analysis(target: str = "peak_flood_level"):
    """Fetches extreme error events and regime diagnostics for hydrological analysis."""
    try:
        return service.get_error_analysis(target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# 3. HISTORICAL DATASET & GAUGE ENDPOINTS
# ==============================================================================

@app.get("/events", response_model=PaginatedEventsResponse, tags=["Historical Events"])
def list_events(
    search: Optional[str] = Query(None, description="Search by EventID, Station, Basin, or State"),
    gauge_id: Optional[str] = Query(None, description="Filter by GaugeID"),
    flood_type: Optional[str] = Query(None, description="Filter by 'Flood' or 'Severe Flood'"),
    basin: Optional[str] = Query(None, description="Filter by river basin"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Lists historical flood events with pagination and search for retrospective replay."""
    offset = (page - 1) * page_size
    total, items = service.list_events(
        search=search, gauge_id=gauge_id, flood_type=flood_type,
        basin=basin, limit=page_size, offset=offset
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items
    }


@app.get("/events/{event_id}", tags=["Historical Events"])
def get_event_detail(event_id: str):
    """Retrieves full predictor and target values for a specific historical event."""
    event = service.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"EventID '{event_id}' not found.")
    
    # Filter out NaNs for clean JSON serialization
    clean_event = {}
    for k, v in event.items():
        if isinstance(v, float) and (v != v or str(v) == "nan"):
            clean_event[k] = None
        else:
            clean_event[k] = v
    return clean_event


@app.get("/gauges", response_model=List[GaugeSummary], tags=["Gauges"])
def list_gauges():
    """Lists all available river monitoring gauges with representative watershed characteristics."""
    return service.list_gauges()


@app.get("/gauges/{gauge_id}", tags=["Gauges"])
def get_gauge_detail(gauge_id: str):
    """Retrieves detailed catchment and baseline features for a specific gauge."""
    gauge = service.get_gauge(gauge_id)
    if not gauge:
        raise HTTPException(status_code=404, detail=f"GaugeID '{gauge_id}' not found.")
    return gauge


# ==============================================================================
# 4. PREDICTION ENDPOINTS
# ==============================================================================

def _record_prediction(
    db: Session,
    pred_type: str,
    mode: str,
    event_id: Optional[str],
    gauge_id: Optional[str],
    model_name: str,
    predicted_val: Optional[float],
    actual_val: Optional[float],
    residual: Optional[float],
    unit: Optional[str],
    result_details: Optional[Dict[str, Any]],
    inputs_dict: Optional[Dict[str, Any]]
):
    """Helper function to record predictions and their inputs into PostgreSQL / database."""
    try:
        rec = PredictionRecord(
            prediction_type=pred_type,
            mode=mode,
            event_id=event_id,
            gauge_id=gauge_id,
            model_name=model_name,
            predicted_value=predicted_val,
            actual_value=actual_val,
            residual=residual,
            unit=unit,
            result_details=json.dumps(result_details) if result_details else None
        )
        db.add(rec)
        db.flush()

        if inputs_dict:
            # Store up to top 30 key features to keep records concise
            stored_count = 0
            for k, v in inputs_dict.items():
                if v is not None and stored_count < 30:
                    inp = PredictionInput(
                        prediction_id=rec.id,
                        feature_name=k,
                        feature_value=str(v)
                    )
                    db.add(inp)
                    stored_count += 1

        db.commit()
    except Exception as e:
        logger.error("Failed to record prediction to database: %s", e)
        db.rollback()


@app.post("/predict/flood-level", response_model=SinglePredictionResponse, tags=["Prediction"])
def predict_flood_level(req: PredictionRequest, db: Session = Depends(get_db)):
    """Predicts Peak Flood Level (meters) using the controlled Gradient Boosting Regressor pipeline."""
    try:
        res = service.predict_regression("peak_flood_level", event_id=req.event_id, features=req.features, gauge_id=req.gauge_id)
        _record_prediction(
            db, "peak_flood_level", res["mode"], res["event_id"], res["gauge_id"],
            res["model_used"], res["predicted_value"], res["actual_value"], res["residual"],
            res["unit"], None, res.get("inputs_used")
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/discharge", response_model=SinglePredictionResponse, tags=["Prediction"])
def predict_discharge(req: PredictionRequest, db: Session = Depends(get_db)):
    """Predicts Peak Discharge Q (cumec) using the controlled Gradient Boosting Regressor pipeline."""
    try:
        res = service.predict_regression("peak_discharge", event_id=req.event_id, features=req.features, gauge_id=req.gauge_id)
        _record_prediction(
            db, "peak_discharge", res["mode"], res["event_id"], res["gauge_id"],
            res["model_used"], res["predicted_value"], res["actual_value"], res["residual"],
            res["unit"], None, res.get("inputs_used")
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/volume", response_model=SinglePredictionResponse, tags=["Prediction"])
def predict_volume(req: PredictionRequest, db: Session = Depends(get_db)):
    """Predicts Flood Volume (cumec) using the controlled Gradient Boosting Regressor pipeline."""
    try:
        res = service.predict_regression("flood_volume", event_id=req.event_id, features=req.features, gauge_id=req.gauge_id)
        _record_prediction(
            db, "flood_volume", res["mode"], res["event_id"], res["gauge_id"],
            res["model_used"], res["predicted_value"], res["actual_value"], res["residual"],
            res["unit"], None, res.get("inputs_used")
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/flood-type", response_model=ClassificationPredictionResponse, tags=["Prediction"])
def predict_flood_type(req: PredictionRequest, db: Session = Depends(get_db)):
    """Classifies flood event into 'Flood' vs 'Severe Flood' using the Gradient Boosting Classifier."""
    try:
        res = service.predict_classification(event_id=req.event_id, features=req.features, gauge_id=req.gauge_id)
        _record_prediction(
            db, "flood_type", res["mode"], res["event_id"], None,
            res["model_used"], None, None, None,
            "category", res, res.get("inputs_used")
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/all", response_model=MultiPredictionResponse, tags=["Prediction"])
def predict_all_targets(req: PredictionRequest, db: Session = Depends(get_db)):
    """Runs simultaneous prediction across Peak Flood Level, Peak Discharge, Flood Volume, and Flood Type."""
    try:
        res = service.predict_all(event_id=req.event_id, features=req.features, gauge_id=req.gauge_id)
        
        # Record combined prediction
        details = {
            "flood_level": res["flood_level"]["predicted_value"],
            "discharge": res["discharge"]["predicted_value"],
            "volume": res["volume"]["predicted_value"],
            "flood_type": res["flood_type"]["predicted_class"] if res.get("flood_type") else None
        }
        _record_prediction(
            db, "all", res["mode"], res["event_id"], res["gauge_id"],
            "Ensemble / Multi-Target Suite", None, None, None,
            "multi", details, None
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_model(req: ExplainRequest):
    """Provides SHAP feature attribution and direction for the specified regression model."""
    try:
        return service.get_shap_explanation(req.target, top_n=req.top_n)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# 5. PREDICTION HISTORY ENDPOINTS
# ==============================================================================

@app.get("/history", response_model=List[PredictionHistoryItem], tags=["Prediction History"])
def get_prediction_history(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """Retrieves saved prediction logs from the PostgreSQL / SQLite database."""
    records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
    results = []
    for r in records:
        details = None
        if r.result_details:
            try:
                details = json.loads(r.result_details)
            except Exception:
                details = {"raw": r.result_details}

        results.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else "",
            "prediction_type": r.prediction_type,
            "mode": r.mode,
            "event_id": r.event_id,
            "gauge_id": r.gauge_id,
            "model_name": r.model_name,
            "predicted_value": r.predicted_value,
            "actual_value": r.actual_value,
            "residual": r.residual,
            "unit": r.unit,
            "result_details": details
        })
    return results


@app.delete("/history/{history_id}", tags=["Prediction History"])
def delete_prediction_history(history_id: int, db: Session = Depends(get_db)):
    """Deletes a specific prediction history record."""
    rec = db.query(PredictionRecord).filter(PredictionRecord.id == history_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    db.delete(rec)
    db.commit()
    return {"status": "success", "message": f"Deleted record {history_id}"}
