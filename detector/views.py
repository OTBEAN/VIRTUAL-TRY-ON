import os
import json
import logging
import random
from datetime import datetime
import requests
from PIL import Image
import io
import base64
import tempfile

import cv2
import numpy as np
import torch
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO

from .models import ClothingItem

# Configure logging
logger = logging.getLogger(__name__)

# Updated VirtualTryOnSystem class with pose-aware clothing adjustment
class VirtualTryOnSystem:
    def __init__(self):
        self.media_root = settings.MEDIA_ROOT
        self.media_url = settings.MEDIA_URL
        self.result_dir = os.path.join(self.media_root, 'tryon_results')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Initialize YOLO models
        try:
            logger.info("Loading enhanced YOLO models...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.pose_model = YOLO("yolov8m-pose.pt").to(device)
            self.segmentation_model = YOLO("yolov8m-seg.pt").to(device)
            logger.info("Enhanced YOLO models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO models: {str(e)}", exc_info=True)
            self.pose_model = YOLO("yolov8m-pose.pt")
            self.segmentation_model = YOLO("yolov8m-seg.pt")
        
        # Initialize advanced model endpoints
        self.oot_diffusion_url = "http://localhost:8001/process"
        self.dci_vton_url = "http://localhost:8002/process"
        self.schp_url = "http://localhost:8003/segment"
        self.densepose_url = "http://localhost:8004/process"

    def call_external_model(self, url, image_data, params=None):
        """Helper method to call external models"""
        try:
            files = {'image': image_data}
            response = requests.post(url, files=files, data=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {url}: {str(e)}")
            return None

    def get_schp_segmentation(self, image):
        """Get human segmentation using SCHP"""
        try:
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Could not encode image")
            
            image_bytes = encoded_image.tobytes()
            response = self.call_external_model(self.schp_url, image_bytes)
            
            if response and response.status_code == 200:
                segmentation_data = response.json()
                mask = np.frombuffer(base64.b64decode(segmentation_data['mask']), np.uint8)
                mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
                return mask
            return None
        except Exception as e:
            logger.error(f"SCHP segmentation failed: {str(e)}")
            return None

    def get_densepose(self, image):
        """Get DensePose estimation"""
        try:
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Could not encode image")
            
            image_bytes = encoded_image.tobytes()
            response = self.call_external_model(self.densepose_url, image_bytes)
            
            if response and response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"DensePose estimation failed: {str(e)}")
            return None

    def generate_oot_diffusion_mask(self, clothing_img, person_img):
        """Generate clothing mask using OOTDiffusion"""
        try:
            success1, encoded_clothing = cv2.imencode('.png', clothing_img)
            success2, encoded_person = cv2.imencode('.png', person_img)
            
            if not success1 or not success2:
                raise ValueError("Could not encode images")
            
            files = {
                'clothing_image': encoded_clothing.tobytes(),
                'person_image': encoded_person.tobytes()
            }
            
            response = requests.post(self.oot_diffusion_url, files=files, timeout=60)
            response.raise_for_status()
            
            result_data = response.json()
            mask_bytes = base64.b64decode(result_data['mask'])
            mask = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
            
            return mask
        except Exception as e:
            logger.error(f"OOTDiffusion mask generation failed: {str(e)}")
            return None

    def apply_dci_vton(self, person_img, clothing_img, densepose_data, segmentation_mask):
        """Apply virtual try-on using DCI-VTON"""
        try:
            success1, encoded_person = cv2.imencode('.png', person_img)
            success2, encoded_clothing = cv2.imencode('.png', clothing_img)
            
            if not success1 or not success2:
                raise ValueError("Could not encode images")
            
            files = {
                'person_image': encoded_person.tobytes(),
                'clothing_image': encoded_clothing.tobytes(),
                'densepose_data': json.dumps(densepose_data),
                'segmentation_mask': base64.b64encode(segmentation_mask.tobytes()).decode('utf-8')
            }
            
            response = requests.post(self.dci_vton_url, files=files, timeout=60)
            response.raise_for_status()
            
            result_data = response.json()
            result_bytes = base64.b64decode(result_data['result_image'])
            result_img = np.frombuffer(result_bytes, np.uint8)
            result_img = cv2.imdecode(result_img, cv2.IMREAD_COLOR)
            
            return result_img
        except Exception as e:
            logger.error(f"DCI-VTON processing failed: {str(e)}")
            return None

    def blend_images_advanced(self, person_img, clothing_img, clothing_type):
        """Advanced blending using all integrated models"""
        try:
            segmentation_mask = self.get_schp_segmentation(person_img)
            if segmentation_mask is None:
                logger.warning("SCHP segmentation failed, using fallback")
                return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)
            
            densepose_data = self.get_densepose(person_img)
            if densepose_data is None:
                logger.warning("DensePose estimation failed, using fallback")
                return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)
            
            clothing_mask = self.generate_oot_diffusion_mask(clothing_img, person_img)
            if clothing_mask is None:
                logger.warning("OOTDiffusion mask generation failed, using fallback")
                return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)
            
            result_img = self.apply_dci_vton(person_img, clothing_img, densepose_data, segmentation_mask)
            if result_img is None:
                logger.warning("DCI-VTON processing failed, using fallback")
                return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)
            
            return result_img
            
        except Exception as e:
            logger.error(f"Advanced blending failed: {str(e)}")
            return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)

    # Existing methods from the original class (maintained for compatibility)
    def process_image(self, image_file):
        """Process uploaded image file with enhanced quality"""
        try:
            image_file.seek(0)
            img_array = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode image")
            
            height, width = img.shape[:2]
            max_dimension = 1000
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
            return img
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}", exc_info=True)
            raise

    def get_enhanced_keypoints(self, image, min_confidence=0.3):
        """Get keypoints with confidence filtering and validation"""
        try:
            height, width = image.shape[:2]
            if height > 640 or width > 640:
                scale = 640 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                resized_img = image
                
            results = self.pose_model(resized_img, verbose=False, conf=0.5)
            if not results or len(results[0].keypoints) == 0:
                logger.warning("No keypoints detected")
                return None

            keypoints = results[0].keypoints.xy.cpu().numpy()
            confidences = results[0].keypoints.conf.cpu().numpy()

            if len(keypoints) == 0:
                return None
                
            person_keypoints = keypoints[0]
            person_confidences = confidences[0]

            if height != resized_img.shape[0] or width != resized_img.shape[1]:
                scale_x = width / resized_img.shape[1]
                scale_y = height / resized_img.shape[0]
                person_keypoints[:, 0] *= scale_x
                person_keypoints[:, 1] *= scale_y

            valid_keypoints = []
            for i, (kp, conf) in enumerate(zip(person_keypoints, person_confidences)):
                if conf >= min_confidence:
                    valid_keypoints.append((i, kp))
                else:
                    valid_keypoints.append((i, None))

            return valid_keypoints
        except Exception as e:
            logger.error(f"Enhanced keypoint detection failed: {str(e)}", exc_info=True)
            return None

    def _get_keypoints(self, image):
        """Internal method to get raw keypoints"""
        keypoints_data = self.get_enhanced_keypoints(image, min_confidence=0.3)
        if keypoints_data is None:
            return None
            
        keypoints = []
        for idx, kp in keypoints_data:
            if kp is not None:
                keypoints.append(kp)
            else:
                keypoints.append([0, 0])
                
        return np.array(keypoints)

    def get_body_landmarks(self, image):
        """Get detailed body landmarks for precise clothing placement"""
        keypoints = self._get_keypoints(image)
        if keypoints is None or len(keypoints) < 17:
            return None
            
        landmarks = {
            'nose': keypoints[0],
            'left_eye': keypoints[1],
            'right_eye': keypoints[2],
            'left_ear': keypoints[3],
            'right_ear': keypoints[4],
            'left_shoulder': keypoints[5],
            'right_shoulder': keypoints[6],
            'left_elbow': keypoints[7],
            'right_elbow': keypoints[8],
            'left_wrist': keypoints[9],
            'right_wrist': keypoints[10],
            'left_hip': keypoints[11],
            'right_hip': keypoints[12],
            'left_knee': keypoints[13],
            'right_knee': keypoints[14],
            'left_ankle': keypoints[15],
            'right_ankle': keypoints[16]
        }
        
        landmarks['shoulder_center'] = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
        landmarks['hip_center'] = (landmarks['left_hip'] + landmarks['right_hip']) / 2
        landmarks['torso_height'] = np.linalg.norm(landmarks['shoulder_center'] - landmarks['hip_center'])
        
        return landmarks

    def calculate_clothing_placement(self, landmarks, clothing_type, clothing_size):
        """Calculate precise clothing placement based on body landmarks"""
        if landmarks is None:
            return None
            
        placement = {
            'type': clothing_type,
            'width': 0,
            'height': 0,
            'position_x': 0,
            'position_y': 0,
            'rotation': 0,
            'scale': 1.0
        }
        
        if clothing_type == 'top':
            shoulder_width = np.linalg.norm(landmarks['left_shoulder'] - landmarks['right_shoulder'])
            torso_height = landmarks['torso_height']
            
            placement['width'] = shoulder_width * 1.2
            placement['height'] = torso_height * 1.1
            placement['position_x'] = landmarks['shoulder_center'][0] - placement['width'] / 2
            placement['position_y'] = landmarks['shoulder_center'][1] - placement['height'] * 0.1
            
            shoulder_angle = np.arctan2(
                landmarks['right_shoulder'][1] - landmarks['left_shoulder'][1],
                landmarks['right_shoulder'][0] - landmarks['left_shoulder'][0]
            )
            placement['rotation'] = np.degrees(shoulder_angle)
            
        elif clothing_type == 'bottom':
            hip_width = np.linalg.norm(landmarks['left_hip'] - landmarks['right_hip'])
            leg_length = np.linalg.norm(landmarks['hip_center'] - landmarks['left_ankle'])
            
            placement['width'] = hip_width * 1.15
            placement['height'] = leg_length * 0.7
            placement['position_x'] = landmarks['hip_center'][0] - placement['width'] / 2
            placement['position_y'] = landmarks['hip_center'][1]
            
            hip_angle = np.arctan2(
                landmarks['right_hip'][1] - landmarks['left_hip'][1],
                landmarks['right_hip'][0] - landmarks['left_hip'][0]
            )
            placement['rotation'] = np.degrees(hip_angle)
            
        return placement

    def warp_clothing_to_pose(self, clothing_img, landmarks, clothing_type):
        """Warp clothing to match body pose using thin plate spline"""
        try:
            if landmarks is None:
                return clothing_img
                
            h, w = clothing_img.shape[:2]
            src_points = np.array([
                [0, 0],
                [w-1, 0],
                [w-1, h-1],
                [0, h-1],
                [w//2, 0],
                [w//2, h-1],
                [0, h//2],
                [w-1, h//2]
            ], dtype=np.float32)
            
            if clothing_type == 'top':
                dst_points = np.array([
                    landmarks['left_shoulder'],
                    landmarks['right_shoulder'],
                    landmarks['right_hip'],
                    landmarks['left_hip'],
                    landmarks['shoulder_center'],
                    landmarks['hip_center'],
                    landmarks['left_elbow'],
                    landmarks['right_elbow']
                ], dtype=np.float32)
                
            elif clothing_type == 'bottom':
                dst_points = np.array([
                    landmarks['left_hip'],
                    landmarks['right_hip'],
                    landmarks['right_ankle'],
                    landmarks['left_ankle'],
                    landmarks['hip_center'],
                    [(landmarks['left_ankle'][0] + landmarks['right_ankle'][0])/2, 
                     (landmarks['left_ankle'][1] + landmarks['right_ankle'][1])/2],
                    landmarks['left_knee'],
                    landmarks['right_knee']
                ], dtype=np.float32)
            
            valid_indices = []
            valid_dst_points = []
            valid_src_points = []
            
            for i, point in enumerate(dst_points):
                if not np.any(np.isnan(point)) and not np.any(np.isinf(point)):
                    valid_indices.append(i)
                    valid_dst_points.append(point)
                    valid_src_points.append(src_points[i])
            
            if len(valid_indices) < 4:
                logger.warning("Not enough valid points for TPS warping")
                return clothing_img
                
            valid_src_points = np.array(valid_src_points, dtype=np.float32)
            valid_dst_points = np.array(valid_dst_points, dtype=np.float32)
            
            tps = cv2.createThinPlateSplineShapeTransformer()
            
            valid_src_points = valid_src_points.reshape(1, -1, 2)
            valid_dst_points = valid_dst_points.reshape(1, -1, 2)
            
            matches = [cv2.DMatch(i, i, 0) for i in range(len(valid_indices))]
            
            tps.estimateTransformation(valid_dst_points, valid_src_points, matches)
            
            warped_img = tps.warpImage(clothing_img)
            
            return warped_img
            
        except Exception as e:
            logger.error(f"Pose warping failed: {str(e)}")
            return clothing_img

    def apply_clothing_with_pose(self, person_img, clothing_img, clothing_type):
        """Apply clothing to person with pose-aware transformation"""
        try:
            # Get body landmarks
            landmarks = self.get_body_landmarks(person_img)
            if landmarks is None:
                logger.warning("No landmarks detected, using fallback placement")
                return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)
            
            # Warp clothing to match pose
            warped_clothing = self.warp_clothing_to_pose(clothing_img, landmarks, clothing_type)
            
            # Calculate placement
            placement = self.calculate_clothing_placement(landmarks, clothing_type, warped_clothing.shape)
            if placement is None:
                return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)
            
            # Resize clothing based on calculated dimensions
            target_size = (int(placement['width']), int(placement['height']))
            resized_clothing = cv2.resize(warped_clothing, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Create mask from alpha channel
            if resized_clothing.shape[2] == 4:
                clothing_mask = resized_clothing[:, :, 3] / 255.0
                clothing_rgb = resized_clothing[:, :, :3]
            else:
                clothing_mask = np.ones(resized_clothing.shape[:2])
                clothing_rgb = resized_clothing
            
            # Calculate position
            y_start = int(placement['position_y'])
            x_start = int(placement['position_x'])
            y_end = y_start + resized_clothing.shape[0]
            x_end = x_start + resized_clothing.shape[1]
            
            # Ensure within bounds
            y_start = max(0, y_start)
            x_start = max(0, x_start)
            y_end = min(person_img.shape[0], y_end)
            x_end = min(person_img.shape[1], x_end)
            
            # Adjust clothing size if needed
            if y_end - y_start != resized_clothing.shape[0] or x_end - x_start != resized_clothing.shape[1]:
                clothing_rgb = clothing_rgb[:y_end-y_start, :x_end-x_start]
                clothing_mask = clothing_mask[:y_end-y_start, :x_end-x_start]
            
            # For dresses, apply additional processing to ensure proper fit
            if clothing_type == 'dress':
                # Create a more refined mask for dresses
                clothing_mask = self.refine_dress_mask(clothing_mask, person_img, y_start, y_end, x_start, x_end)
            
            # Blend clothing onto person
            result = person_img.copy().astype(np.float32)
            
            for c in range(3):
                result[y_start:y_end, x_start:x_end, c] = (
                    clothing_rgb[:, :, c] * clothing_mask +
                    result[y_start:y_end, x_start:x_end, c] * (1 - clothing_mask)
                )
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Pose-aware clothing application failed: {str(e)}")
            return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)

    def refine_dress_mask(self, original_mask, person_img, y_start, y_end, x_start, x_end):
        """Refine the mask for dresses to ensure better blending"""
        try:
            # Get the region of interest from the person image
            roi = person_img[y_start:y_end, x_start:x_end]
            
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi
            
            # Apply adaptive thresholding to detect body contours
            thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a refined mask
            refined_mask = original_mask.copy()
            
            # If we found contours, use them to refine the mask
            if contours:
                # Create a mask from the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                contour_mask = np.zeros_like(roi_gray)
                cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
                
                # Blend the original mask with the contour mask
                refined_mask = cv2.bitwise_and(original_mask, contour_mask)
            
            return refined_mask
            
        except Exception as e:
            logger.error(f"Dress mask refinement failed: {str(e)}")
            return original_mask

    def apply_clothing_fallback(self, person_img, clothing_img, clothing_type):
        """Fallback method for when pose detection fails"""
        try:
            h, w = person_img.shape[:2]
            ch, cw = clothing_img.shape[:2]
            
            if clothing_type == 'top':
                y_start = int(h * 0.2)
                scale = min(w * 0.6 / cw, h * 0.4 / ch)
            else:
                y_start = int(h * 0.5)
                scale = min(w * 0.7 / cw, h * 0.5 / ch)
            
            new_width = int(cw * scale)
            new_height = int(ch * scale)
            resized_clothing = cv2.resize(clothing_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            x_start = (w - new_width) // 2
            
            if resized_clothing.shape[2] == 4:
                mask = resized_clothing[:, :, 3] / 255.0
                clothing_rgb = resized_clothing[:, :, :3]
            else:
                mask = np.ones((new_height, new_width))
                clothing_rgb = resized_clothing
            
            result = person_img.copy().astype(np.float32)
            y_end = min(h, y_start + new_height)
            x_end = min(w, x_start + new_width)
            
            for c in range(3):
                result[y_start:y_end, x_start:x_end, c] = (
                    clothing_rgb[:y_end-y_start, :x_end-x_start, c] * mask[:y_end-y_start, :x_end-x_start] +
                    result[y_start:y_end, x_start:x_end, c] * (1 - mask[:y_end-y_start, :x_end-x_start])
                )
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Fallback clothing application failed: {str(e)}")
            return person_img

    def get_body_measurements(self, image):
        """Enhanced with better validation using YOLO pose keypoints"""
        keypoints = self._get_keypoints(image)
        if keypoints is None or len(keypoints) < 17:
            logger.error("No or insufficient keypoints detected in image")
            return None

        try:
            left_shoulder = keypoints[5] if np.any(keypoints[5]) else None
            right_shoulder = keypoints[6] if np.any(keypoints[6]) else None
            left_hip = keypoints[11] if np.any(keypoints[11]) else None
            right_hip = keypoints[12] if np.any(keypoints[12]) else None
            
            if left_shoulder is None or right_shoulder is None:
                logger.error("Shoulder keypoints missing")
                return None
                
            if left_hip is None or right_hip is None:
                logger.error("Hip keypoints missing")
                return None
            
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            hip_width = np.linalg.norm(left_hip - right_hip)
            
            if shoulder_width < 20 or hip_width < 20:
                logger.error(f"Unrealistic measurements - shoulder: {shoulder_width}, hip: {hip_width}")
                return None
                
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, 
                              (left_shoulder[1] + right_shoulder[1]) / 2]
            hip_center = [(left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2]
            
            torso_height = abs(shoulder_center[1] - hip_center[1])
            
            return {
                "shoulder_width": int(shoulder_width),
                "hip_width": int(hip_width),
                "shoulder_height": int(shoulder_center[1]),
                "shoulder_center": int(shoulder_center[0]),
                "hip_height": int(hip_center[1]),
                "hip_center": int(hip_center[0]),
                "torso_height": int(torso_height)
            }
        except Exception as e:
            logger.error(f"Measurement calculation error: {str(e)}")
            return None

    def get_body_segmentation(self, image):
        """Get precise body segmentation with edge refinement"""
        try:
            results = self.segmentation_model(image, verbose=False)
            if not results or not results[0].masks:
                return None
            
            mask = results[0].masks[0].data[0].cpu().numpy()
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            return (mask > 0.5).ast(np.uint8) * 255
        except Exception as e:
            logger.error(f"Segmentation failed: {str(e)}")
            return None

    def segment_clothing(self, clothing_img):
        """Segment clothing from background using thresholding"""
        try:
            if len(clothing_img.shape) == 3:
                gray = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = clothing_img
            
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
        except Exception as e:
            logger.error(f"Clothing segmentation failed: {str(e)}")
            raise

    def process_clothing(self, clothing_path, clothing_type, measurements):
        """Improved clothing processing with segmentation and scaling"""
        try:
            logger.info(f"Processing clothing from path: {clothing_path}")
            
            # Check if file exists
            if not os.path.exists(clothing_path):
                raise ValueError(f"Clothing image file does not exist: {clothing_path}")
                
            clothing_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)
            if clothing_img is None:
                raise ValueError("Could not read clothing image")

            logger.info(f"Original clothing image shape: {clothing_img.shape}")

            # Segment clothing from background
            mask = self.segment_clothing(clothing_img)
            
            # Convert to RGBA with mask as alpha channel
            if clothing_img.shape[2] == 3:
                clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2BGRA)
            
            clothing_img[:, :, 3] = mask  # Set alpha channel
            
            # Calculate scaling based on body measurements and clothing type
            if clothing_type == 'top' or clothing_type == 'dress':
                # For tops and dresses, use shoulder width with margin
                target_width = measurements['shoulder_width'] * 1.3  # 30% margin
                scale_factor = target_width / clothing_img.shape[1]
                
                # For dresses, also consider length
                if clothing_type == 'dress':
                    # Estimate dress length based on torso height
                    torso_height = measurements['torso_height']
                    target_height = torso_height * 2.2  # Adjust based on typical dress length
                    height_scale = target_height / clothing_img.shape[0]
                    # Use the larger scale factor to ensure proper fit
                    scale_factor = max(scale_factor, height_scale)
            else:
                # For bottoms, use hip width with margin
                target_width = measurements['hip_width'] * 1.2  # 20% margin
                scale_factor = target_width / clothing_img.shape[1]

            # Apply scaling
            new_size = (int(clothing_img.shape[1] * scale_factor), 
                       int(clothing_img.shape[0] * scale_factor))
            
            # High-quality resize
            resized = cv2.resize(
                clothing_img, 
                new_size, 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            logger.info(f"Clothing resized from {clothing_img.shape} to {resized.shape}")
            return resized
            
        except Exception as e:
            logger.error(f"Clothing processing failed: {str(e)}", exc_info=True)
            raise
    
    def apply_shadows_and_highlights(self, person_img, clothing_area, position, measurements):
        """Add realistic lighting effects with body awareness"""
        try:
            lab = cv2.cvtColor(person_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            kernel_size = int(min(person_img.shape[0], person_img.shape[1]) * 0.05)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            kernel = np.ones((kernel_size, kernel_size), np.float32)/(kernel_size*kernel_size)
            
            shadow_mask = cv2.filter2D(clothing_area.astype(np.float32), -1, kernel) / 255.0
            
            y1, y2 = position[1], position[1] + clothing_area.shape[0]
            x1, x2 = position[0], position[0] + clothing_area.shape[1]
            
            y1, y2 = max(0, y1), min(person_img.shape[0], y2)
            x1, x2 = max(0, x1), min(person_img.shape[1], x2)
            
            shadow_mask = shadow_mask[:y2-y1, :x2-x1]
            
            l_roi = l[y1:y2, x1:x2]
            l_roi[:] = np.clip(l_roi * (1 - shadow_mask * 0.3), 0, 255)
            
            lab = cv2.merge((l,a,b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.error(f"Shadow effect failed: {str(e)}")
            return person_img

    def add_fabric_texture(self, clothing_img):
        """Add realistic fabric texture with noise"""
        try:
            height, width = clothing_img.shape[:2]
            x = np.linspace(0, 5, width)
            y = np.linspace(0, 5, height)
            xv, yv = np.meshgrid(x, y)
            
            noise = np.sin(xv) * np.cos(yv) * 10
            noise = noise.astype(np.uint8)
            
            for c in range(3):
                clothing_img[:,:,c] = cv2.addWeighted(clothing_img[:,:,c], 0.9, noise, 0.1, 0)
            return clothing_img
        except Exception as e:
            logger.error(f"Texture generation failed: {str(e)}")
            return clothing_img

    def blend_images(self, person_img, clothing_img, clothing_type, measurements, segmentation_mask=None):
        """Updated blend_images method using pose-aware approach"""
        try:
            result_img = self.apply_clothing_with_pose(person_img, clothing_img, clothing_type)
            
            result_img = self.apply_shadows_and_highlights(
                result_img, 
                clothing_img[:, :, 3] if clothing_img.shape[2] == 4 else None,
                (0, 0),
                measurements
            )
            
            return result_img
            
        except Exception as e:
            logger.error(f"Blending failed: {str(e)}")
            return self.apply_clothing_fallback(person_img, clothing_img, clothing_type)

    def save_result(self, image):
        """Enhanced result saving with validation and backup"""
        try:
            if image is None:
                raise ValueError("Cannot save None image")
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Image must be 3-channel BGR. Got shape: {image.shape}")

            os.makedirs(self.result_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = str(random.randint(1000, 9999))
            filename = f"result_{timestamp}_{random_suffix}.jpg"
            result_path = os.path.join(self.result_dir, filename)
            
            save_success = cv2.imwrite(
                result_path,
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            
            if not save_success:
                logger.warning("First save attempt failed, trying alternative method")
                temp_path = os.path.join(self.result_dir, f"temp_{filename}")
                cv2.imwrite(temp_path, image)
                if os.path.exists(temp_path):
                    os.rename(temp_path, result_path)
                    save_success = True
            
            if not save_success:
                raise ValueError("All save attempts failed")
                
            if not os.path.exists(result_path):
                raise ValueError("File was not created")
            if os.path.getsize(result_path) == 0:
                os.remove(result_path)
                raise ValueError("Saved file is empty")
                
            logger.info(f"Result saved successfully at {result_path}")
            return os.path.join(self.media_url, 'tryon_results', filename)
            
        except Exception as e:
            logger.error(f"Result saving failed: {str(e)}", exc_info=True)
            
            try:
                backup_dir = os.path.join(self.media_root, 'backup_results')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f"backup_{filename}")
                if cv2.imwrite(backup_path, image):
                    logger.info(f"Saved backup result at {backup_path}")
            except Exception as backup_error:
                logger.error(f"Backup save also failed: {str(backup_error)}")
                
            raise

# Initialize the try-on system
tryon_system = VirtualTryOnSystem()

def get_clothing_items():
    """Get all available clothing items with caching"""
    return ClothingItem.objects.exclude(image__isnull=True).exclude(image='')

@csrf_exempt
def debug_pose(request):
    """Endpoint to return pose detection data for visualization"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            img = tryon_system.process_image(request.FILES['image'])
            landmarks = tryon_system.get_body_landmarks(img)
            
            # Convert landmarks to serializable format
            serializable_landmarks = {}
            if landmarks:
                for name, point in landmarks.items():
                    if point is not None and len(point) == 2:
                        serializable_landmarks[name] = {'x': float(point[0]), 'y': float(point[1])}
            
            return JsonResponse({
                'landmarks': serializable_landmarks,
                'success': True
            })
        except Exception as e:
            return JsonResponse({'error': str(e), 'success': False}, status=400)
    return JsonResponse({'error': 'Invalid request', 'success': False}, status=400)

def home_page(request):
    """Main virtual try-on interface"""
    return render(request, "combined.html", {
        'clothes': get_clothing_items(),
        'media_url': settings.MEDIA_URL
    })

@csrf_exempt
def upload_clothing(request):
    """Handle clothing uploads with validation"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

    try:
        # Validate inputs
        image_file = request.FILES.get('image')
        name = request.POST.get('name', '').strip()
        category = request.POST.get('category', 'other')

        if not image_file:
            return JsonResponse({'status': 'error', 'message': 'No image provided'}, status=400)
        if not name:
            return JsonResponse({'status': 'error', 'message': 'Name is required'}, status=400)
        if image_file.size > 5 * 1024 * 1024:  # 5MB limit
            return JsonResponse({'status': 'error', 'message': 'Image too large (max 5MB)'}, status=400)

        # Save file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clothing/{timestamp}_{image_file.name}"
        filepath = default_storage.save(filename, ContentFile(image_file.read()))

        # Create database record
        clothing = ClothingItem.objects.create(
            name=name,
            category=category,
            image=filepath,
            updated_at=datetime.now()
        )

        return JsonResponse({
            'status': 'success',
            'clothing': {
                'id': clothing.id,
                'name': clothing.name,
                'category': clothing.category,
                'image_url': request.build_absolute_uri(clothing.image.url)
            }
        })

    except Exception as e:
        logger.error(f"Clothing upload failed: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)



@csrf_exempt
def static_tryon(request):
    """Enhanced virtual try-on endpoint with detailed error reporting"""
    debug_data = {
        'timestamp': datetime.now().isoformat(),
        'received_files': bool(request.FILES),
    }
    
    try:
        # Validate request method
        if request.method != 'POST':
            return JsonResponse({'error': 'Method not allowed'}, status=405)

        # Validate inputs
        if not request.FILES.get('image'):
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        clothing_id = request.POST.get('clothing_id')
        
        if not clothing_id:
            return JsonResponse({'error': 'Clothing ID required'}, status=400)

        # Get clothing item
        try:
            clothing = ClothingItem.objects.get(id=clothing_id)
            if not clothing.image:
                return JsonResponse({'error': 'Clothing image missing'}, status=400)
        except ObjectDoesNotExist:
            return JsonResponse({'error': 'Clothing not found'}, status=404)

        # Process person image
        person_img = tryon_system.process_image(image_file)
        if person_img is None:
            return JsonResponse({'error': 'Image processing failed'}, status=400)

        # Determine clothing type
        clothing_type = 'top' if clothing.category in ['top', 'shirt', 't-shirt', 'dress'] else 'bottom'
        
        # Get body measurements
        measurements = tryon_system.get_body_measurements(person_img)
        if not measurements:
            return JsonResponse({'error': 'Could not detect body pose'}, status=400)

        # Process clothing image
        clothing_img = tryon_system.process_clothing(
            clothing.image.path,
            clothing_type,
            measurements
        )
        if clothing_img is None:
            return JsonResponse({'error': 'Clothing processing failed'}, status=400)

        # Use fallback method (more reliable)
        result_img = tryon_system.apply_clothing_fallback(
            person_img, 
            clothing_img, 
            clothing_type
        )
        
        if result_img is None:
            return JsonResponse({'error': 'Outfit blending failed'}, status=400)

        # Save result
        result_url = tryon_system.save_result(result_img)
        
        return JsonResponse({
            'status': 'success',
            'result_image': request.build_absolute_uri(result_url)
        })

    except Exception as e:
        logger.error(f"Virtual try-on failed: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': 'Internal server error',
            'details': str(e)
        }, status=500)
    
@csrf_exempt
def debug_clothing_processing(request):
    """Endpoint to debug clothing processing"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Process clothing image
            clothing_img = tryon_system.process_image(request.FILES['image'])
            
            # Segment clothing
            mask = tryon_system.segment_clothing(clothing_img)
            
            # Convert to base64 for response
            _, buffer = cv2.imencode('.png', mask)
            mask_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JsonResponse({
                'mask': mask_base64,
                'success': True
            })
        except Exception as e:
            return JsonResponse({'error': str(e), 'success': False}, status=400)
    return JsonResponse({'error': 'Invalid request', 'success': False}, status=400)

@csrf_exempt
def debug_pose_detection(request):
    """Endpoint to debug pose detection"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            img = tryon_system.process_image(request.FILES['image'])
            landmarks = tryon_system.get_body_landmarks(img)
            
            # Draw landmarks on image
            debug_img = img.copy()
            if landmarks:
                for name, point in landmarks.items():
                    if point is not None and len(point) == 2:
                        cv2.circle(debug_img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                        cv2.putText(debug_img, name, (int(point[0]) + 5, int(point[1])), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Convert to base64 for response
            _, buffer = cv2.imencode('.png', debug_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JsonResponse({
                'image': img_base64,
                'success': True
            })
        except Exception as e:
            return JsonResponse({'error': str(e), 'success': False}, status=400)
    return JsonResponse({'error': 'Invalid request', 'success': False}, status=400)

def get_clothing(request):
    """API endpoint for clothing items with optimized query"""
    clothes = get_clothing_items()
    return JsonResponse({
        "clothes": [
            {
                "id": item.id,
                "name": item.name,
                "category": item.category,
                "image_url": request.build_absolute_uri(item.image.url)
            }
            for item in clothes
        ]
    })

@csrf_exempt
def test_models(request):
    """Proper JSON test endpoint"""
    try:
        # Simple test that doesn't actually load models
        return JsonResponse({
            'status': 'success',
            'message': 'Test endpoint working',
            'models': {
                'pose': 'available',
                'segmentation': 'available'
            }
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)
    
@csrf_exempt
def delete_clothing(request, item_id):
    if request.method != 'DELETE':
        return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
    
    try:
        clothing = ClothingItem.objects.get(id=item_id)
        clothing.image.delete()  # Delete the image file
        clothing.delete()       # Delete the database record
        return JsonResponse({'status': 'success'})
    except ObjectDoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Clothing not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
def debug_tryon(request):
    """Endpoint to return debug images"""
    if request.method == 'POST':
        try:
            if not request.FILES.get('image'):
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            img = tryon_system.process_image(request.FILES['image'])
            if img is None:
                return JsonResponse({'error': 'Image processing failed'}, status=400)
            
            # Get keypoints for visualization
            keypoints = tryon_system._get_keypoints(img)
            
            # Create debug visualization
            debug_img = img.copy()
            
            if keypoints is not None:
                # Draw keypoints
                for i, point in enumerate(keypoints):
                    if point is not None and len(point) == 2:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(debug_img, (x, y), 5, (0, 255, 0), -1)
                        cv2.putText(debug_img, str(i), (x + 5, y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Convert to JPEG
            success, buf = cv2.imencode('.jpg', debug_img)
            if not success:
                return JsonResponse({'error': 'Image encoding failed'}, status=500)
                
            return HttpResponse(buf.tobytes(), content_type='image/jpeg')
            
        except Exception as e:
            logger.error(f"Debug error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'POST required'}, status=400)