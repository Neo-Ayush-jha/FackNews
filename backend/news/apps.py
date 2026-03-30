from django.apps import AppConfig


class NewsConfig(AppConfig):
    name = 'news'
    
    def ready(self):
        """
        Pre-load the model when Django starts.
        This ensures fast first request response time.
        """
        try:
            from .utils import load_model_background
            load_model_background()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not pre-load model: {e}")
