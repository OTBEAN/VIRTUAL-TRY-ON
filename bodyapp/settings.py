"""
Django settings for bodyapp project.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Security settings
SECRET_KEY = "django-insecure-zii1%bg!^)w(&elkgy_ubo71098t%2fp@+89s&zv@327^wl)87"
DEBUG = True
ALLOWED_HOSTS = ['*']  # Allow all hosts for development

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

     'detector.apps.DetectorConfig',  
]
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'debug.log',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "bodyapp.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, 'templates')],  # Add templates directory
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.media",  # Add media context processor
            ],
        },
    },
]

WSGI_APPLICATION = "bodyapp.wsgi.application"

# Database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(BASE_DIR, "db.sqlite3"),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'detector/static'),
]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Media files (Uploaded clothes and results)
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

# Create required directories if they don't exist
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'tryon_results'), exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'clothes'), exist_ok=True)

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# YOLO model path (ensure this file exists in your project root)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n-pose.pt')