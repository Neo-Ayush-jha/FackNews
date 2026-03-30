from django.conf import settings
from django.db import models

class Prediction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
        null=True,
        blank=True,
    )
    text = models.TextField()
    result = models.CharField(max_length=20)
    confidence = models.FloatField()
    verification_result = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        owner = self.user if self.user else "Anonymous"
        return f"{owner} - {self.result} - {self.confidence}%"
