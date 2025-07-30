import os
import pandas as pd
import joblib
from models.market_forecaster import MarketForecaster
from models.maintenance_predictor import MaintenancePredictor

def initialize_forecaster(market_data, model_path='models/market_forecaster.pkl'):
    """Load or train and save market forecaster model"""
    forecaster = MarketForecaster()
    if os.path.exists(model_path):
        try:
            forecaster.load_model(model_path)
        except Exception as e:
            print(f" Error loading market forecaster model: {e}. Re-training...")
            forecaster.train_ensemble_models(market_data)
            forecaster.save_model(model_path)
    else:
        forecaster.train_ensemble_models(market_data)
        forecaster.save_model(model_path)
    return forecaster

def initialize_maintenance_model(equipment_data, model_path='models/maintenance_model.pkl'):
    """Load or train and save maintenance prediction model"""
    model = MaintenancePredictor()
    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
        except Exception as e:
            print(f" Error loading maintenance model: {e}. Re-training...")
            model.train_anomaly_model(equipment_data)
            model.save_model(model_path)
    else:
        model.train_anomaly_model(equipment_data)
        model.save_model(model_path)
    return model
