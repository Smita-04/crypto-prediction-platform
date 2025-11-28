# Milestone3/prediction_app/tasks.py
import logging
from celery import shared_task
from django.conf import settings
from django.utils import timezone
from datetime import datetime

# Import the actual prediction logic
from prediction_app.ml_models.services.binance_service import (
    predict_live_price as binance_predict_live_price,
    coin_mapping # Needed for display names if not passed directly
)

# Import PredictionHistory model to save results
from .models import PredictionHistory

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def perform_prediction_task(self, user_id, cryptocurrency_symbol, data_type, output_currency, duration):
    """
    Celery task to perform the cryptocurrency prediction.
    """
    logger.info(f"Celery task {self.request.id} started for user {user_id}, crypto {cryptocurrency_symbol}")
    
    try:
        # Call your main prediction service function
        prediction_result_data = binance_predict_live_price(
            coin_symbol_from_frontend=cryptocurrency_symbol,
            data_type=data_type,
            currency_output=output_currency
        )

        if prediction_result_data and prediction_result_data.get('predicted_price_usd') is not None:
            # Retrieve the user object for saving history
            from django.contrib.auth import get_user_model
            User = get_user_model()
            user = User.objects.get(id=user_id)

            # Get the relevant predicted and current price based on output_currency
            predicted_value = prediction_result_data.get(f'predicted_price_{output_currency.lower()}')
            price_at_prediction_time = prediction_result_data.get(f'current_price_{output_currency.lower()}')

            if predicted_value is None or price_at_prediction_time is None:
                logger.error(f"Task {self.request.id}: Predicted value or price at prediction time is None after service call.")
                # Return error info or raise an exception
                return {'status': 'error', 'message': 'Prediction service returned incomplete data.'}

            # Create an interval label using duration (from frontend) and data_type
            history_interval_label = f"{duration} {data_type.capitalize()}"

            # Save prediction to history
            history_entry = PredictionHistory.objects.create(
                user=user,
                cryptocurrency=cryptocurrency_symbol,
                interval=history_interval_label,
                output_currency=output_currency,
                predicted_value=predicted_value,
                price_at_prediction_time=price_at_prediction_time,
                # actual_price and accuracy remain null for now
            )
            logger.info(f"Celery task {self.request.id}: Prediction saved to history (ID: {history_entry.id}).")

            # Prepare the result to be returned by the task
            result_for_frontend = {
                'status': 'success',
                'cryptocurrency': cryptocurrency_symbol,
                'current_price': round(float(price_at_prediction_time), 2),
                'predicted_price': round(float(predicted_value), 2),
                'output_currency': output_currency,
                'history_id': history_entry.id,
                'graphic': prediction_result_data.get('graphic'), # Assuming graphic is base64 string
                # If you later add a confidence, it would be here
                # 'confidence': prediction_result_data.get('confidence', 'N/A')
            }
            return result_for_frontend

        else:
            logger.error(f"Celery task {self.request.id}: Prediction service returned empty or invalid data: {prediction_result_data}")
            return {'status': 'error', 'message': 'Prediction service returned empty or invalid data.'}

    except Exception as e:
        logger.exception(f"Celery task {self.request.id}: An unexpected error occurred during prediction: {e}")
        return {'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}