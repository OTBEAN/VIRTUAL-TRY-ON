from django.apps import AppConfig

class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    # If you have label, make sure it's unique
    label = 'detector'  # Remove this line if present