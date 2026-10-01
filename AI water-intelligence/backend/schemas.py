from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    event_id: Optional[str] = Field(None, description="Optional EventID for historical replay demonstration")
    gauge_id: Optional[str] = Field(None, description="Optional GaugeID for catchment template selection")
    features: Optional[Dict[str, Any]] = Field(None, description="Dictionary of predictor values for scenario prediction")

class SinglePredictionResponse(BaseModel):
    target: str
    target_key: str
    predicted_value: float
    actual_value: Optional[float] = None
    residual: Optional[float] = None
    model_used: str
    unit: str
    mode: str
    event_id: Optional[str] = None
    gauge_id: Optional[str] = None
    disclaimer: str
    timestamp: str

class ClassificationPredictionResponse(BaseModel):
    target: str = "Flood Type"
    target_key: str = "flood_type"
    predicted_class: str
    probability_flood: Optional[float] = None
    probability_severe_flood: Optional[float] = None
    confidence: Optional[float] = None
    actual_class: Optional[str] = None
    model_used: str
    status: str = "operational"
    mode: str
    event_id: Optional[str] = None
    disclaimer: str
    timestamp: str

class MultiPredictionResponse(BaseModel):
    event_id: Optional[str] = None
    gauge_id: Optional[str] = None
    mode: str
    disclaimer: str
    timestamp: str
    flood_level: SinglePredictionResponse
    discharge: SinglePredictionResponse
    volume: SinglePredictionResponse
    flood_type: Optional[ClassificationPredictionResponse] = None

class ExplainRequest(BaseModel):
    target: str = Field("peak_flood_level", description="Target key: peak_flood_level, peak_discharge, or flood_volume")
    top_n: int = Field(15, ge=5, le=50, description="Number of top SHAP features to return")

class ExplainFeature(BaseModel):
    feature: str
    display_name: str
    category: str
    mean_abs_shap: float
    mean_shap: Optional[float] = None
    direction: Optional[str] = None
    positive_contribution_pct: Optional[float] = None
    negative_contribution_pct: Optional[float] = None

class ExplainResponse(BaseModel):
    target: str
    target_key: str
    model_name: str
    top_features: List[ExplainFeature]
    disclaimer: str

class EventSummary(BaseModel):
    EventID: str
    GaugeID: Optional[str] = None
    Station: Optional[str] = None
    Basin: Optional[str] = None
    State: Optional[str] = None
    Start_Date: Optional[str] = None
    Peak_Flood_Level: Optional[float] = None
    Peak_Discharge: Optional[float] = None
    Flood_Volume: Optional[float] = None
    Flood_Type: Optional[str] = None

class PaginatedEventsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[EventSummary]

class GaugeSummary(BaseModel):
    GaugeID: str
    Station: Optional[str] = None
    Basin: Optional[str] = None
    State: Optional[str] = None
    Latitude: Optional[float] = None
    Longitude: Optional[float] = None
    Drainage_Area: Optional[float] = None
    Stream_Order: Optional[float] = None
    Event_Count: int

class AnalyticsSummaryResponse(BaseModel):
    total_events: int
    flood_count: int
    severe_flood_count: int
    total_gauges: int
    integrity_audit_status: str
    integrity_audit_pass: int
    integrity_audit_fail: int
    regression_models: List[Dict[str, Any]]
    classification_models: List[Dict[str, Any]]

class PredictionHistoryItem(BaseModel):
    id: int
    timestamp: str
    prediction_type: str
    mode: str
    event_id: Optional[str] = None
    gauge_id: Optional[str] = None
    model_name: str
    predicted_value: Optional[float] = None
    actual_value: Optional[float] = None
    residual: Optional[float] = None
    unit: Optional[str] = None
    result_details: Optional[Dict[str, Any]] = None
