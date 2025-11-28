# prediction_app/apps.py
from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class PredictionAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction_app'

    def ready(self):
        # This simpler ready method will not try to load models initially
        # We are just checking if the AppConfig class itself can be found.
        logger.info(f"PredictionAppConfig for '{self.name}' is ready (simplified).")
        # You can add a placeholder print to confirm it's reached:
        # print("DEBUG: PredictionAppConfig's ready method executed!")
