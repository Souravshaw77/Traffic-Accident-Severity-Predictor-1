from sqlalchemy import Column, Integer, String, Float, DateTime, func
from db import Base, engine


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())

    day_of_week = Column(String(20), nullable=True)
    weather_conditions = Column(String(50), nullable=True)
    light_conditions = Column(String(50), nullable=True)
    num_vehicles = Column(Integer, nullable=True)
    num_casualties = Column(Integer, nullable=True)
    hour = Column(Integer, nullable=True)

    predicted_severity = Column(String(50), nullable=False)
    prob_fatal = Column(Float, nullable=True)
    prob_serious = Column(Float, nullable=True)
    prob_slight = Column(Float, nullable=True)

    model_version = Column(String(20), nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)
