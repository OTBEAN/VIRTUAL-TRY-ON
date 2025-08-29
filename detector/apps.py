from django.apps import AppConfig
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Import SCHP model here

app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process with SCHP model
    # segmentation_mask = schp_model.process(image)
    
    # For now, return a placeholder
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Convert mask to base64
    _, buffer = cv2.imencode('.png', mask)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'mask': mask_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    # If you have label, make sure it's unique
    label = 'detector'  # Remove this line if present


