# urls.py
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views

urlpatterns = [
    path('', views.home_page, name='home'),
    path('static-tryon/', views.static_tryon, name='static_tryon'),
    path('upload-clothing/', views.upload_clothing, name='upload_clothing'),
    path('delete-clothing/<int:item_id>/', views.delete_clothing, name='delete_clothing'),
    path('get-clothing/', views.get_clothing, name='get_clothing'),
    path('test-models/', views.test_models, name='test_models'),
    path('debug-pose/', views.debug_pose, name='debug_pose'),
    path('debug-tryon/', views.debug_tryon, name='debug_tryon'),
    path('debug-clothing-processing/', views.debug_clothing_processing, name='debug_clothing_processing'),
    path('debug-pose-detection/', views.debug_pose_detection, name='debug_pose_detection'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)