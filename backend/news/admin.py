from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
	list_display = ("id", "user", "result", "confidence", "created_at")
	list_filter = ("result", "created_at", "user")
	search_fields = ("text", "user__username", "user__email")
	ordering = ("-created_at",)
