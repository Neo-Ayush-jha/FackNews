import logging
import os
import sys

from django.apps import AppConfig


logger = logging.getLogger(__name__)


class NewsConfig(AppConfig):
    name = 'news'
    
    def ready(self):
        """
        Keep Django startup fast. Model warmup is opt-in because loading
        torch/transformers during runserver startup is slow on most laptops.
        """
        preload_enabled = os.getenv("ENABLE_MODEL_PRELOAD", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not preload_enabled:
            return

        if "runserver" not in sys.argv:
            return

        # Django's autoreloader starts a parent and a child process. Only warm
        # the model in the child process when preload is explicitly enabled.
        if os.environ.get("RUN_MAIN") != "true":
            return

        try:
            from .utils import load_model_background
            load_model_background()
        except Exception as e:
            logger.warning(f"Could not pre-load model: {e}")
