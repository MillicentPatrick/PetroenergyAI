import os
import pandas as pd
import joblib
import logging
from datetime import datetime
from models.market_forecaster import MarketForecaster
from models.maintenance_predictor import MaintenancePredictor

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PetroEnergyAI')

class ModelPipeline:
    """Centralized pipeline for model management with version control"""
    
    MODEL_VERSION = "1.0.0"
    COMPATIBLE_VERSIONS = ["1.0.0", "0.9.0"]
    
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_model_path(self, model_name):
        """Generate versioned model path"""
        return os.path.join(self.models_dir, f"{model_name}_v{self.MODEL_VERSION}.pkl")

    def initialize_forecaster(self, market_data):
        """
        Initialize market forecaster with automatic version handling
        
        Args:
            market_data (pd.DataFrame): Preprocessed market data
            
        Returns:
            MarketForecaster: Initialized and trained forecaster
        """
        model_path = self._get_model_path("market_forecaster")
        forecaster = MarketForecaster()
        
        if self._try_load_model(forecaster, model_path):
            logger.info("Market forecaster loaded successfully")
        else:
            logger.info("Training new market forecaster model")
            start_time = datetime.now()
            forecaster.train_ensemble_models(market_data)
            forecaster.save_model(model_path)
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
        return forecaster

    def initialize_maintenance_model(self, equipment_data):
        """
        Initialize maintenance predictor with automatic version handling
        
        Args:
            equipment_data (pd.DataFrame): Equipment sensor data
            
        Returns:
            MaintenancePredictor: Initialized and trained model
        """
        model_path = self._get_model_path("maintenance_predictor")
        predictor = MaintenancePredictor()
        
        if self._try_load_model(predictor, model_path):
            logger.info("Maintenance predictor loaded successfully")
        else:
            logger.info("Training new maintenance predictor model")
            start_time = datetime.now()
            predictor.train_anomaly_model(equipment_data)
            predictor.save_model(model_path)
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
        return predictor

    def _try_load_model(self, model_instance, model_path):
        """
        Attempt to load and validate a model with version checking
        
        Args:
            model_instance: Model object to load into
            model_path: Path to model file
            
        Returns:
            bool: True if load was successful
        """
        if not os.path.exists(model_path):
            logger.warning(f"No model found at {model_path}")
            return False
            
        try:
            # Check model version compatibility
            with open(model_path, 'rb') as f:
                model_data = joblib.load(f)
                
                if isinstance(model_data, dict) and 'version' in model_data:
                    if model_data['version'] not in self.COMPATIBLE_VERSIONS:
                        logger.warning(f"Incompatible model version {model_data['version']}")
                        return False
                        
            # Load the actual model
            model_instance.load_model(model_path)
            
            # Validate model
            if not model_instance.is_trained():
                logger.warning("Loaded model failed validation")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False

    def cleanup_old_models(self):
        """Remove outdated model versions"""
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                file_version = filename.split('_v')[-1].replace('.pkl', '')
                if file_version not in self.COMPATIBLE_VERSIONS:
                    old_path = os.path.join(self.models_dir, filename)
                    os.remove(old_path)
                    logger.info(f"Removed outdated model: {filename}")

def initialize_forecaster(market_data):
    """Legacy compatibility function"""
    return ModelPipeline().initialize_forecaster(market_data)

def initialize_maintenance_model(equipment_data):
    """Legacy compatibility function"""
    return ModelPipeline().initialize_maintenance_model(equipment_data)
