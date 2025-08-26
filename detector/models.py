# detector/models.py
from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

def validate_image(image):
    if not image:
        raise ValidationError(_("No image was provided"))
    if image.size > 5 * 1024 * 1024:  # 5MB limit
        raise ValidationError(_("Image file too large ( > 5MB )"))

# detector/models.py
from django.db import models

class ClothingItem(models.Model):
    CATEGORY_CHOICES = [
        ('top', 'Top'),
        ('bottom', 'Bottom'),
        ('dress', 'Dress'),
        ('other', 'Other'),
    ]
    
    name = models.CharField(max_length=100)
    category = models.CharField(max_length=10, choices=CATEGORY_CHOICES)
    image = models.ImageField(upload_to='clothing/')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)  # This will auto-update

    def __str__(self):
        return self.name

    def clean(self):
        if not self.image:
            raise ValidationError({'image': 'Image is required'})

    class Meta:
        verbose_name = "Clothing Item"
        verbose_name_plural = "Clothing Items"
        ordering = ['-created_at']