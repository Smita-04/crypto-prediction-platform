from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    cryptocurrency = models.CharField(max_length=100) # e.g., 'Bitcoin' or 'BTC'
    interval = models.CharField(max_length=50) # e.g., '1 Hour', '1 Day'
    output_currency = models.CharField(max_length=10, default='USD') # 'USD' or 'INR'
    predicted_value = models.DecimalField(max_digits=18, decimal_places=8)
    prediction_date = models.DateTimeField(default=timezone.now)

    # --- NEW FIELDS FOR ACTUAL PRICE AND ACCURACY ---
    # These will initially be null and can be updated by a separate process
    price_at_prediction_time = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    
    class Meta:
        verbose_name_plural = "Prediction History"
        ordering = ['-prediction_date']

    def __str__(self):
        return f"{self.user.username}'s prediction for {self.cryptocurrency} ({self.interval}) on {self.prediction_date.strftime('%Y-%m-%d')}"
