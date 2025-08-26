from django.contrib import admin
from django.urls import path, include

from django.views.generic.base import RedirectView


# Define urlpatterns list
urlpatterns = [
    path('admin/', admin.site.urls),
  
    path('', include('detector.urls')),  # Make sure 'detector' is your app name
        # Handle Chrome DevTools requests
    path('.well-known/appspecific/com.chrome.devtools.json', 
         RedirectView.as_view(url='/static/empty.json')),

   
] 

# For development only - to serve media files
from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

