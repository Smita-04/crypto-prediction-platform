from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import datetime, timedelta

# Standard library imports
import json
import csv
import os
import logging
import io
import base64

# Third-party data processing and ML imports (now only for graph generation)
import matplotlib.pyplot as plt

# Your Django app's model
from .models import PredictionHistory

# Import Celery task
from .tasks import perform_prediction_task 

# These imports are now primarily used by the Celery task directly,
# but might be kept if you have other synchronous uses for them.
# For the make_prediction view, we don't directly call binance_predict_live_price anymore.
from prediction_app.ml_models.services.binance_service import (
    # predict_live_price as binance_predict_live_price, # This is now called in the Celery task
    KLINE_INTERVAL_1HOUR, # Might be used elsewhere if you keep live price fetching
    KLINE_INTERVAL_1DAY,  # Might be used elsewhere if you keep live price fetching
    KLINE_INTERVAL_1MINUTE, # Might be used elsewhere if you keep live price fetching
    get_live_klines,      # Might be used elsewhere if you keep live price fetching
    coin_mapping,
)


logger = logging.getLogger(__name__)


@login_required
def welcome_view(request):
    """
    Renders the welcome page for logged-in users, displaying only cryptocurrency names and symbols initially.
    Live prices will be fetched dynamically via AJAX when 'Refresh Prices' is clicked.
    """
    display_crypto_data = []
    for symbol, info in coin_mapping.items():
        display_crypto_data.append({
            'name': info["fullname"],
            'symbol': symbol,
            'icon': symbol.lower(), # Assuming icon names like 'btc.png'
            'price': "N/A", # Initially no live price, will be fetched by JS refresh
            'change': "N/A" # Initially no change
        })

    context = {
        'username': request.user.username,
        'crypto_data_json': json.dumps(display_crypto_data), # Still pass for JS to handle refresh
        'live_crypto_data_count': len(display_crypto_data),
        'user': request.user
    }
    return render(request, 'prediction_app/welcome.html', context)


@login_required
def prediction_form(request):
    """
    Renders the HTML form for cryptocurrency prediction.
    """
    cryptocurrencies = []
    for symbol, info in coin_mapping.items():
        cryptocurrencies.append({'code': symbol, 'name': info["fullname"]})

    context = {
        'cryptocurrencies': cryptocurrencies,
    }
    return render(request, 'prediction_app/prediction_form.html', context)


@login_required
@require_POST # Ensure only POST requests are processed
@csrf_protect # Ensure CSRF token is checked (frontend is sending it now)
def make_prediction(request):
    """
    API endpoint to receive prediction requests (POST) from JavaScript,
    schedule a Celery task, and return a task ID immediately.
    """
    cryptocurrency = None # Initialize for logging purposes in exception handler
    interval_type_from_frontend = None # Initialize to match new key from frontend
    output_currency = None # Initialize for logging
    duration = None # Initialize for logging

    try:
        data = json.loads(request.body.decode('utf-8')) # Decode and parse JSON

        cryptocurrency = data.get('cryptocurrency')
        interval_type_from_frontend = data.get('timeframe_type')
        output_currency = data.get('output_currency', 'USD')
        duration = data.get('duration')

        # --- Basic Validation ---
        if not all([cryptocurrency, interval_type_from_frontend, output_currency, duration is not None]):
            logger.error(f"Missing required parameters in request: cryptocurrency={cryptocurrency}, timeframe_type={interval_type_from_frontend}, output_currency={output_currency}, duration={duration}")
            return JsonResponse({'status': 'error', 'message': 'Missing required prediction parameters.'}, status=400)
        
        if interval_type_from_frontend not in ['hourly', 'daily']:
            logger.error(f"Invalid timeframe_type received: {interval_type_from_frontend}")
            return JsonResponse({'status': 'error', 'message': 'Invalid timeframe type. Must be \"hourly\" or \"daily\".'}, status=400)

        if not isinstance(duration, int) or duration <= 0:
            logger.error(f"Invalid duration received: {duration}")
            return JsonResponse({'status': 'error', 'message': 'Duration must be a positive integer.'}, status=400)

        logger.info(f"Scheduling prediction task for: Crypto={cryptocurrency}, Type={interval_type_from_frontend}, Output={output_currency}, Duration={duration} by user {request.user.id}")

        # --- Schedule the Celery task ---
        task = perform_prediction_task.delay(
            request.user.id,
            cryptocurrency,
            interval_type_from_frontend,
            output_currency,
            duration
        )

        # Return the task ID immediately to the frontend
        return JsonResponse({'status': 'processing', 'task_id': task.id}, status=202) # 202 Accepted

    except json.JSONDecodeError:
        logger.error("Invalid JSON body received in POST request.", exc_info=True)
        return JsonResponse({'status': 'error', 'message': 'Invalid request data (JSON format error).'}, status=400)
    except Exception as e:
        logger.exception(f"An unexpected error occurred while scheduling prediction task: {e}")
        return JsonResponse({'status': 'error', 'message': f'An unexpected server error occurred: {str(e)}.'}, status=500)


@login_required
def prediction_status(request, task_id):
    """
    API endpoint to check the status of a Celery prediction task.
    """
    task = perform_prediction_task.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response_data = {'status': 'pending'}
    elif task.state == 'PROGRESS':
        response_data = {'status': 'progress', 'current_step': task.info.get('current_step', 'N/A')}
    elif task.state == 'SUCCESS':
        # Task completed successfully, retrieve result
        prediction_result = task.result # This is the dictionary returned by perform_prediction_task
        
        # Check if the task itself returned an error status
        if prediction_result and prediction_result.get('status') == 'error':
            logger.error(f"Celery task {task_id} reported an internal error: {prediction_result.get('message')}")
            return JsonResponse({
                'status': 'error',
                'message': prediction_result.get('message', 'Prediction task failed internally.')
            }, status=500)
        
        # Prediction was successful, pass data to frontend
        # The frontend will likely redirect based on this, or update UI directly
        logger.info(f"Celery task {task_id} completed successfully.")

        # Store the graphic and other relevant data in session for prediction_result_view
        # We fetch the PredictionHistory entry using the history_id from the task result
        history_id = prediction_result.get('history_id')
        if history_id:
            try:
                history_entry = PredictionHistory.objects.get(id=history_id)
                # Reconstruct session data for prediction_result_view, now using saved graphic
                request.session['prediction_result'] = {
                    'cryptocurrency': history_entry.cryptocurrency,
                    'interval': history_entry.interval,
                    'output_currency': history_entry.output_currency,
                    'predicted_value': float(history_entry.predicted_value),
                    'current_price': float(history_entry.price_at_prediction_time) if history_entry.price_at_prediction_time is not None else 'N/A',
                    # graphic is generated and stored in the Celery task, then retrieved or passed
                    # For a simple approach, we can directly pass the base64 graphic string if the task returns it.
                    'graphic': prediction_result.get('graphic'),
                    'history_id': history_entry.id,
                }
            except PredictionHistory.DoesNotExist:
                logger.error(f"Celery task {task_id} reported history_id {history_id} but record not found.")
                # Fallback to returning data directly without session if history entry not found
                pass 

        response_data = {
            'status': 'success',
            'cryptocurrency': prediction_result.get('cryptocurrency'),
            'current_price': prediction_result.get('current_price'),
            'predicted_price': prediction_result.get('predicted_price'),
            'output_currency': prediction_result.get('output_currency'),
            'history_id': prediction_result.get('history_id'),
            'graphic': prediction_result.get('graphic'), # Pass graphic back if returned by task
        }
    elif task.state == 'FAILURE':
        # Task failed, retrieve exception details
        logger.error(f"Celery task {task_id} failed with exception: {task.info}", exc_info=True)
        response_data = {'status': 'error', 'message': 'Prediction task failed. Please check server logs.'}
    else:
        response_data = {'status': 'unknown'}

    return JsonResponse(response_data)


@login_required
def prediction_result_view(request):
    """
    Displays the result of the last prediction, including the graph.
    This view now relies on data stored in the session by the prediction_status endpoint.
    """
    prediction_data = request.session.pop('prediction_result', None)
    if not prediction_data:
        messages.warning(request, 'No recent prediction found. Please make a prediction first.')
        return redirect('prediction_form')

    return render(request, 'prediction_app/prediction_result.html', {'prediction_data': prediction_data})


@login_required
def history_view(request):
    """
    Displays a list of all past predictions for the current user, with filtering.
    """
    predictions_queryset = PredictionHistory.objects.filter(user=request.user)

    filter_coin = request.GET.get('coin', '')
    filter_date_from = request.GET.get('date_from', '')
    filter_date_to = request.GET.get('date_to', '')

    if filter_coin:
        predictions_queryset = predictions_queryset.filter(cryptocurrency=filter_coin)
    
    if filter_date_from:
        try:
            date_from_obj = datetime.strptime(filter_date_from, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            predictions_queryset = predictions_queryset.filter(prediction_date__gte=date_from_obj)
        except ValueError:
            messages.error(request, "Invalid 'Date From' format. Please use YYYY-MM-DD.")
            logger.warning(f"Invalid 'Date From' format received: {filter_date_from}")

    if filter_date_to:
        try:
            date_to_obj = datetime.strptime(filter_date_to, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            predictions_queryset = predictions_queryset.filter(prediction_date__lte=date_to_obj)
        except ValueError:
            messages.error(request, "Invalid 'Date To' format. Please use YYYY-MM-DD.")
            logger.warning(f"Invalid 'Date To' format received: {filter_date_to}")
    
    predictions_queryset = predictions_queryset.order_by('-prediction_date')

    # Add currency symbol and formatted prices to each prediction for display
    predictions_with_display_data = []
    for prediction in predictions_queryset:
        display_currency_symbol = '$' if prediction.output_currency == 'USD' else '₹'
        
        # Format values for display
        formatted_predicted_value = f"{display_currency_symbol}{prediction.predicted_value:.2f}"
        
        # New: Format price_at_prediction_time for display
        formatted_price_at_prediction_time = f"{display_currency_symbol}{prediction.price_at_prediction_time:.2f}" if prediction.price_at_prediction_time is not None else "N/A"


        predictions_with_display_data.append({
            'id': prediction.id,
            'prediction_date': prediction.prediction_date.astimezone(timezone.get_current_timezone()),
            'cryptocurrency': prediction.cryptocurrency,
            'interval': prediction.interval,
            'output_currency': prediction.output_currency,
            'formatted_predicted_value': formatted_predicted_value,
            'formatted_price_at_prediction_time': formatted_price_at_prediction_time, # New formatted field
            'raw_predicted_value': prediction.predicted_value,
            'raw_price_at_prediction_time': prediction.price_at_prediction_time,
        })


    cryptocurrencies_for_filter = []
    for symbol, info in coin_mapping.items():
        cryptocurrencies_for_filter.append({'code': symbol, 'name': info["fullname"]})

    context = {
        'predictions': predictions_with_display_data, # Pass the new list
        'cryptocurrencies': cryptocurrencies_for_filter,
        'filter_coin': filter_coin,
        'filter_date_from': filter_date_from,
        'filter_date_to': filter_date_to,
    }
    return render(request, 'prediction_app/history.html', context)


@login_required
def download_history_report(request):
    """
    Generates and returns a CSV file of the user's prediction history.
    """
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_history.csv"'

    writer = csv.writer(response)
    # Updated CSV header
    writer.writerow(['Date', 'Cryptocurrency', 'Interval', 'Output Currency', 'Price at Prediction Time', 'Predicted Value',])

    predictions_queryset = PredictionHistory.objects.filter(user=request.user).order_by('-prediction_date')

    filter_coin = request.GET.get('coin', '')
    filter_date_from = request.GET.get('date_from', '')
    filter_date_to = request.GET.get('date_to', '')

    if filter_coin:
        predictions_queryset = predictions_queryset.filter(cryptocurrency=filter_coin)
    
    if filter_date_from:
        try:
            date_from_obj = datetime.strptime(filter_date_from, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            predictions_queryset = predictions_queryset.filter(prediction_date__gte=date_from_obj)
        except ValueError:
            logger.warning(f"Invalid 'Date From' format for download: {filter_date_from}")
    
    if filter_date_to:
        try:
            date_to_obj = datetime.strptime(filter_date_to, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            predictions_queryset = predictions_queryset.filter(prediction_date__lte=date_to_obj)
        except ValueError:
            logger.warning(f"Invalid 'Date To' format for download: {filter_date_to}")

    for prediction in predictions_queryset:
        # Format values for CSV
        price_at_prediction_time_str = f"{prediction.price_at_prediction_time:.2f}" if prediction.price_at_prediction_time is not None else "N/A"
        

        writer.writerow([
            prediction.prediction_date.astimezone(timezone.get_current_timezone()).strftime('%Y-%m-%d %H:%M:%S'),
            prediction.cryptocurrency,
            prediction.interval,
            prediction.output_currency,
            price_at_prediction_time_str, # New field
            f"{prediction.predicted_value:.2f}",
        ])

    return response


@login_required
@csrf_protect # Ensure CSRF is checked for POST requests
def live_prices_api_view(request):
    """
    API endpoint to fetch and return live market prices for the dashboard from Binance.
    """
    if request.method == 'GET':
        live_crypto_data = []
        for symbol, info in coin_mapping.items():
            binance_symbol = f"{symbol}USDT"
            data = get_live_klines(binance_symbol, KLINE_INTERVAL_1MINUTE, 1) # Fetch latest 1-min kline
            
            current_price = "N/A"
            if data is not None and not data.empty:
                current_price = float(data['Close'].iloc[-1]) # Convert to float
            
            live_crypto_data.append({
                'name': info["fullname"],
                'symbol': symbol,
                'price': current_price,
                'change': "N/A" # Calculating 24hr change requires fetching more data (e.g., 24hr klines)
            })
        
        return JsonResponse({"status": "success", "live_prices": live_crypto_data})
    else:
        return JsonResponse({"status": "error", "message": "Only GET requests are allowed for this API endpoint."}, status=405)


@login_required
@require_POST # Ensure only POST requests are processed
@csrf_protect # Ensure CSRF is checked for POST requests
def delete_prediction_record(request, prediction_id):
    """
    API endpoint to delete a specific prediction record for the logged-in user.
    """
    try:
        prediction = PredictionHistory.objects.get(id=prediction_id, user=request.user)
        prediction.delete()
        messages.success(request, "Prediction record deleted successfully.")
        return JsonResponse({'status': 'success', 'message': f'Record {prediction_id} deleted.'})
    except PredictionHistory.DoesNotExist:
        logger.warning(f"Attempt to delete non-existent or unauthorized prediction ID: {prediction_id} by user {request.user.id}")
        return JsonResponse({'status': 'error', 'message': 'Record not found or you do not have permission to delete it.'}, status=404)
    except Exception as e:
        logger.error(f"Error deleting prediction {prediction_id} for user {request.user.id}: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error deleting record: {str(e)}'}, status=500)


@login_required
@require_POST # Ensure only POST requests are processed
@csrf_protect # Ensure CSRF is checked for POST requests
def clear_all_history(request):
    """
    API endpoint to clear all prediction history for the logged-in user.
    """
    try:
        PredictionHistory.objects.filter(user=request.user).delete()
        messages.success(request, "All prediction history cleared successfully.")
        return JsonResponse({'status': 'success', 'message': 'All history cleared.'})
    except Exception as e:
        logger.error(f"Error clearing all history for user {request.user.id}: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': f'Error clearing history: {str(e)}'}, status=500)
