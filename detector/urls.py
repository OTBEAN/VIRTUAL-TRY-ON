from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Main page
    path('', views.home_page, name='home'),
    
    # API endpoints
    path('static-tryon/', views.static_tryon, name='static_tryon'),
    path('upload-clothing/', views.upload_clothing, name='upload_clothing'),
    path('get-clothing/', views.get_clothing, name='get_clothing'),
    path('delete-clothing/<int:item_id>/', views.delete_clothing, name='delete_clothing'),
    path('debug-tryon/', views.debug_tryon, name='debug-tryon'),
    path('test-models/', views.test_models, name='test_models'),  # Add this line
     path('debug-pose/', views.debug_pose, name='debug_pose'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)