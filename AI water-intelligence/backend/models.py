from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prediction_type = Column(String(50), nullable=False)  # peak_flood_level, peak_discharge, flood_volume, flood_type, all
    mode = Column(String(50), nullable=False)  # historical_replay, scenario
    event_id = Column(String(100), nullable=True, index=True)
    gauge_id = Column(String(50), nullable=True, index=True)
    model_name = Column(String(100), nullable=False)
    predicted_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    residual = Column(Float, nullable=True)
    unit = Column(String(50), nullable=True)
    result_details = Column(Text, nullable=True)  # JSON string for multi-prediction or classification details

    inputs = relationship("PredictionInput", back_populates="prediction", cascade="all, delete-orphan")

class PredictionInput(Base):
    __tablename__ = "prediction_inputs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Text, nullable=True)

    prediction = relationship("PredictionRecord", back_populates="inputs")
