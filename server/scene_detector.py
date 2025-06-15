import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import json
import urllib.request
import os

@dataclass
class DetectedRegion:
    """Structure to hold detected region information"""
    region_type: str  # 'polygon' or 'line'
    coordinates: List[Tuple[int, int]]
    confidence: float
    suggested_alert: str
    alert_config: Dict[str, Any]

@dataclass
class SceneAnalysisResult:
    """Structure to hold complete scene analysis result"""
    scene_type: str
    confidence: float
    regions: List[DetectedRegion]
    raw_detections: List[Dict]
    places365_predictions: List[Tuple[str, float]]  # Top-5 predictions from Places365

class SceneDetector:
    """Main scene detection and region suggestion system with Places365 integration"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', use_places365: bool = True):
        """
        Initialize the scene detector
        
        Args:
            model_path: Path to YOLO model (will download if not exists)
            use_places365: Whether to use Places365 for scene classification
        """
        print("Loading YOLO model...")
        self.yolo_model = YOLO(model_path)
        
        # Initialize Places365 if requested
        self.use_places365 = use_places365
        self.places365_model = None
        self.places365_classes = None
        self.places365_transform = None
        
        if use_places365:
            self._load_places365()
        
        # Enhanced scene rules with Places365 mapping
        self.scene_rules = {
            'car_park': {
                'required_objects': ['car'],
                'optional_objects': ['truck', 'bus'],
                'spatial_pattern': 'multiple_vehicles_arranged',
                'places365_scenes': ['parking_lot', 'parking_garage', 'street', 'driveway']
            },
            'entrance': {
                'required_objects': ['door'],
                'optional_objects': ['person'],
                'spatial_pattern': 'door_near_edge',
                'places365_scenes': ['doorway', 'entrance_hall', 'building_facade', 'lobby']
            },
            'office_desk': {
                'required_objects': ['chair'],
                'optional_objects': ['laptop', 'keyboard', 'mouse', 'monitor', 'tvmonitor'],
                'spatial_pattern': 'desk_furniture_cluster',
                'places365_scenes': ['office', 'office_cubicles', 'computer_room', 'conference_room']
            },
            'general_door': {
                'required_objects': ['door'],
                'optional_objects': [],
                'spatial_pattern': 'single_door',
                'places365_scenes': ['corridor', 'hallway', 'doorway']
            },
            'warehouse': {
                'required_objects': [],
                'optional_objects': ['person', 'truck'],
                'spatial_pattern': 'industrial_space',
                'places365_scenes': ['warehouse', 'storage_room', 'loading_dock']
            },
            'retail': {
                'required_objects': [],
                'optional_objects': ['person'],
                'spatial_pattern': 'retail_space',
                'places365_scenes': ['shop', 'store', 'supermarket', 'shopping_mall']
            }
        }
        
        # Alert type recommendations for each scene
        self.alert_recommendations = {
            'car_park': {
                'alert_type': 'vehicle_infringement',
                'config': {
                    'detection_type': 'vehicle',
                    'action': 'count_and_alert',
                    'time_threshold': 300,  # seconds
                    'confidence_threshold': 0.7
                }
            },
            'entrance': {
                'alert_type': 'line_crossing',
                'config': {
                    'detection_type': 'person',
                    'action': 'count_crossing',
                    'direction': 'bidirectional',
                    'confidence_threshold': 0.6
                }
            },
            'office_desk': {
                'alert_type': 'region_intrusion',
                'config': {
                    'detection_type': 'person',
                    'action': 'intrusion_alert',
                    'time_threshold': 60,
                    'confidence_threshold': 0.6
                }
            },
            'general_door': {
                'alert_type': 'access_control',
                'config': {
                    'detection_type': 'person',
                    'action': 'access_log',
                    'direction': 'bidirectional',
                    'confidence_threshold': 0.7
                }
            },
            'warehouse': {
                'alert_type': 'safety_monitoring',
                'config': {
                    'detection_type': 'person',
                    'action': 'ppe_detection',
                    'confidence_threshold': 0.7
                }
            },
            'retail': {
                'alert_type': 'customer_analytics',
                'config': {
                    'detection_type': 'person',
                    'action': 'count_and_track',
                    'confidence_threshold': 0.6
                }
            }
        }
    
    def _load_places365(self):
        """Load Places365 model and categories"""
        print("Loading Places365 model...")
        
        # Download categories file if not exists
        if not os.path.exists('categories_places365.txt'):
            print("Downloading Places365 categories...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt',
                'categories_places365.txt'
            )
        
        # Load categories
        classes = []
        with open('categories_places365.txt') as f:
            for line in f:
                classes.append(line.strip().split(' ')[0][3:])
        self.places365_classes = tuple(classes)
        
        # Load the pre-trained Places365 model
        # Using ResNet50 backbone - you can also use ResNet18 for faster inference
        arch = 'resnet50'
        model_file = f'{arch}_places365.pth.tar'
        
        if not os.path.exists(model_file):
            print(f"Downloading Places365 {arch} model...")
            model_url = f'http://places2.csail.mit.edu/models_places365/{model_file}'
            urllib.request.urlretrieve(model_url, model_file)
        
        # Create model
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        self.places365_model = model
        
        # Image transformer
        self.places365_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Places365 model loaded successfully!")
    
    def classify_with_places365(self, image: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify scene using Places365
        
        Args:
            image: Input image as numpy array (RGB)
            top_k: Number of top predictions to return
            
        Returns:
            List of (scene_name, confidence) tuples
        """
        if not self.use_places365 or self.places365_model is None:
            return []
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Transform and predict
        input_tensor = self.places365_transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.places365_model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get top k predictions
        top_probs, top_classes = probs[0].topk(top_k)
        
        predictions = []
        for i in range(top_k):
            scene_name = self.places365_classes[top_classes[i]]
            confidence = top_probs[i].item()
            predictions.append((scene_name, confidence))
        
        return predictions
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the image using YOLO
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected objects with coordinates and confidence
        """
        results = self.yolo_model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    })
        
        return detections
    
    def classify_scene(self, detections: List[Dict], image_shape: Tuple[int, int], 
                      places365_predictions: List[Tuple[str, float]] = None) -> Tuple[str, float]:
        """
        Classify the scene based on detected objects, spatial patterns, and Places365
        
        Args:
            detections: List of detected objects
            image_shape: (height, width) of the image
            places365_predictions: Optional Places365 predictions
            
        Returns:
            Tuple of (scene_type, confidence)
        """
        detected_classes = [det['class_name'] for det in detections]
        height, width = image_shape
        
        scene_scores = {}
        
        # First, check Places365 predictions if available
        places365_boost = {}
        if places365_predictions:
            for scene_type, rules in self.scene_rules.items():
                if 'places365_scenes' in rules:
                    for places_scene, places_conf in places365_predictions:
                        if places_scene in rules['places365_scenes']:
                            # Boost score based on Places365 confidence
                            places365_boost[scene_type] = places365_boost.get(scene_type, 0) + places_conf * 0.5
        
        # Combine object detection and Places365 scores
        for scene_type, rules in self.scene_rules.items():
            score = 0
            
            # Add Places365 boost if available
            if scene_type in places365_boost:
                score += places365_boost[scene_type]
            
            # Check required objects
            required_found = sum(1 for obj in rules['required_objects'] if obj in detected_classes)
            if required_found == 0 and len(rules['required_objects']) > 0:
                # If Places365 gave high confidence, still consider the scene
                if score < 0.3:
                    scene_scores[scene_type] = score
                    continue
            
            score += required_found * 0.3
            
            # Check optional objects
            optional_found = sum(1 for obj in rules['optional_objects'] if obj in detected_classes)
            score += optional_found * 0.15
            
            # Spatial pattern analysis
            if scene_type == 'car_park':
                car_detections = [det for det in detections if det['class_name'] in ['car', 'truck', 'bus']]
                if len(car_detections) >= 2:
                    score += 0.2  # Multiple vehicles suggest parking lot
                    
            elif scene_type == 'entrance' or scene_type == 'general_door':
                door_detections = [det for det in detections if det['class_name'] == 'door']
                for door in door_detections:
                    x1, y1, x2, y2 = door['bbox']
                    # Check if door is near image edge (typical for entrance)
                    if (x1 < width * 0.1 or x2 > width * 0.9 or 
                        y1 < height * 0.1 or y2 > height * 0.9):
                        if scene_type == 'entrance':
                            score += 0.2
                    else:
                        if scene_type == 'general_door':
                            score += 0.2
                            
            elif scene_type == 'office_desk':
                chair_detections = [det for det in detections if det['class_name'] == 'chair']
                monitor_detections = [det for det in detections if det['class_name'] in ['tvmonitor', 'laptop']]
                if len(chair_detections) >= 1:
                    score += 0.2
                if len(monitor_detections) >= 1:
                    score += 0.1
            
            scene_scores[scene_type] = min(score, 1.0)
        
        # If Places365 detected a scene with high confidence that we don't have rules for,
        # still report it
        if places365_predictions and places365_predictions[0][1] > 0.5:
            top_places_scene = places365_predictions[0][0]
            if max(scene_scores.values()) < 0.3:
                # No strong match in our rules, use Places365 directly
                return f"places365_{top_places_scene}", places365_predictions[0][1]
        
        # Return the scene with highest score
        if not scene_scores or max(scene_scores.values()) < 0.2:
            return 'unknown', 0.0
        
        best_scene = max(scene_scores.items(), key=lambda x: x[1])
        return best_scene[0], best_scene[1]
    
    def detect_parking_spaces(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Detect parking spaces using white line detection and contour analysis
        
        Args:
            image: Input image
            detections: Existing object detections
            
        Returns:
            List of detected parking space regions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect white lines (parking bay markings)
        # Use threshold to isolate white lines
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Use morphology to clean up the lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of white line segments
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find parking bay lines (rectangular shapes)
        parking_bay_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for rectangular shapes (parking bays)
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Parking spaces are typically wider than tall in this view
                    if 0.8 < aspect_ratio < 3.0 and w > 80 and h > 40:
                        parking_bay_contours.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.9,
                            'type': 'parking_space',
                            'contour': contour
                        })
        
        # If contour detection doesn't find enough spaces, use line-based detection
        if len(parking_bay_contours) < 4:
            parking_bay_contours = self.detect_parking_by_lines(image)
        
        # If still not enough, fall back to smart grid
        if len(parking_bay_contours) < 4:
            parking_bay_contours = self.generate_smart_grid_parking_spaces(image, detections)
        
        return parking_bay_contours
    
    def detect_parking_by_lines(self, image: np.ndarray) -> List[Dict]:
        """
        Detect parking spaces using Hough line detection focused on parking bay lines
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection tuned for white lines on asphalt
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        parking_spaces = []
        
        if lines is not None:
            # Group lines by orientation and position
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 30:  # Filter short lines
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # Classify as horizontal or vertical
                    if abs(angle) < 20 or abs(angle) > 160:  # Horizontal-ish
                        horizontal_lines.append((x1, y1, x2, y2, (y1+y2)/2))
                    elif 70 < abs(angle) < 110:  # Vertical-ish
                        vertical_lines.append((x1, y1, x2, y2, (x1+x2)/2))
            
            # Sort vertical lines by x-coordinate
            vertical_lines.sort(key=lambda x: x[4])
            
            # Try to find parking spaces between vertical lines
            height, width = image.shape[:2]
            
            if len(vertical_lines) >= 2:
                for i in range(len(vertical_lines) - 1):
                    x1_avg = vertical_lines[i][4]
                    x2_avg = vertical_lines[i + 1][4]
                    
                    if abs(x2_avg - x1_avg) > 60:  # Minimum parking space width
                        # Find corresponding horizontal boundaries
                        y_top = int(height * 0.4)  # Adjust based on typical parking lot layout
                        y_bottom = int(height * 0.8)
                        
                        parking_spaces.append({
                            'bbox': [int(x1_avg), y_top, int(x2_avg), y_bottom],
                            'confidence': 0.8,
                            'type': 'parking_space'
                        })
        
        return parking_spaces
    
    def generate_smart_grid_parking_spaces(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Generate parking spaces using a smart grid approach based on detected cars and visible lines
        """
        height, width = image.shape[:2]
        parking_spaces = []
        
        # Get car detections to help guide parking space placement
        car_detections = [det for det in detections if det['class_name'] in ['car', 'truck', 'bus']]
        
        if len(car_detections) >= 2:
            # Use car positions to estimate parking space layout
            car_centers = [(det['center'][0], det['center'][1]) for det in car_detections]
            
            # Sort cars by x-position to find spacing pattern
            car_centers.sort(key=lambda x: x[0])
            
            # Calculate average spacing between cars
            if len(car_centers) >= 2:
                spacings = []
                for i in range(1, len(car_centers)):
                    spacing = car_centers[i][0] - car_centers[i-1][0]
                    if 60 < spacing < 300:  # Reasonable parking space width
                        spacings.append(spacing)
                
                if spacings:
                    avg_spacing = sum(spacings) / len(spacings)
                    
                    # Generate grid based on car positions and spacing
                    leftmost_car = min(car_centers, key=lambda x: x[0])
                    rightmost_car = max(car_centers, key=lambda x: x[0])
                    
                    # Extend beyond car positions to catch empty spaces
                    start_x = max(50, leftmost_car[0] - avg_spacing)
                    end_x = min(width - 50, rightmost_car[0] + avg_spacing)
                    
                    # Estimate parking space boundaries based on image geometry
                    avg_car_y = sum([pos[1] for pos in car_centers]) / len(car_centers)
                    space_height = 80  # Typical parking space depth
                    y_top = max(50, int(avg_car_y - space_height/2))
                    y_bottom = min(height - 50, int(avg_car_y + space_height/2))
                    
                    # Generate parking spaces
                    current_x = start_x
                    space_count = 0
                    
                    while current_x < end_x and space_count < 10:  # Limit to reasonable number
                        x_left = int(current_x)
                        x_right = int(current_x + avg_spacing)
                        
                        if x_right <= width - 50:
                            parking_spaces.append({
                                'bbox': [x_left, y_top, x_right, y_bottom],
                                'confidence': 0.7,
                                'type': 'parking_space'
                            })
                            space_count += 1
                        
                        current_x += avg_spacing
        
        # Fallback to simple grid if car-based detection fails
        if len(parking_spaces) < 3:
            parking_spaces = self.generate_simple_grid_parking_spaces(image)
        
        return parking_spaces
    
    def generate_simple_grid_parking_spaces(self, image: np.ndarray) -> List[Dict]:
        """
        Generate a simple grid of parking spaces as last resort
        """
        height, width = image.shape[:2]
        parking_spaces = []
        
        # Define parking area based on typical surveillance camera view
        parking_start_y = int(height * 0.45)  # Start lower to avoid background
        parking_end_y = int(height * 0.75)    # Don't go to bottom edge
        parking_start_x = int(width * 0.1)
        parking_end_x = int(width * 0.9)
        
        # Estimate parking space count and size
        estimated_spaces = min(7, max(4, (parking_end_x - parking_start_x) // 120))
        space_width = (parking_end_x - parking_start_x) // estimated_spaces
        
        for i in range(estimated_spaces):
            x1 = parking_start_x + i * space_width
            x2 = parking_start_x + (i + 1) * space_width - 5  # Small gap between spaces
            
            parking_spaces.append({
                'bbox': [x1, parking_start_y, x2, parking_end_y],
                'confidence': 0.6,
                'type': 'parking_space'
            })
        
        return parking_spaces

    def generate_regions(self, scene_type: str, detections: List[Dict], 
                        image_rgb: np.ndarray) -> List[DetectedRegion]:
        """
        Generate suggested regions based on scene type and detections
        
        Args:
            scene_type: Classified scene type
            detections: List of detected objects
            image_rgb: Input image for additional analysis
            
        Returns:
            List of suggested regions with normalized coordinates (0-1)
        """
        regions = []
        height, width = image_rgb.shape[:2]
        
        # Handle Places365-only scenes
        if scene_type.startswith('places365_'):
            # For scenes detected only by Places365, generate generic monitoring regions
            # based on detected objects
            person_detections = [det for det in detections if det['class_name'] == 'person']
            
            # Create monitoring regions around people
            for person in person_detections:
                x1, y1, x2, y2 = person['bbox']
                margin = 50
                
                polygon_coords_pixel = [
                    (max(0, x1 - margin), max(0, y1 - margin)),
                    (min(width, x2 + margin), max(0, y1 - margin)),
                    (min(width, x2 + margin), min(height, y2 + margin)),
                    (max(0, x1 - margin), min(height, y2 + margin))
                ]
                
                polygon_coords_normalized = [
                    (x / width, y / height) for x, y in polygon_coords_pixel
                ]
                
                regions.append(DetectedRegion(
                    region_type='polygon',
                    coordinates=polygon_coords_normalized,
                    confidence=person['confidence'],
                    suggested_alert='person_detection',
                    alert_config={
                        'detection_type': 'person',
                        'action': 'monitor',
                        'confidence_threshold': 0.6
                    }
                ))
            
            return regions
        
        if scene_type == 'car_park':
            # 1. Detect ALL parking spaces (not just where cars are)
            parking_spaces = self.detect_parking_spaces(image_rgb, detections)
            
            for i, space in enumerate(parking_spaces):
                x1, y1, x2, y2 = space['bbox']
                
                # Calculate parking bay positioning as you specified:
                # Top 2 points: at 1/3 of the bay height from top
                # Bottom 2 points: at the bay edges (bottom)
                bay_width = x2 - x1
                bay_height = y2 - y1
                
                # Top edge at 1/3 of bay height
                top_y = y1 + bay_height // 3
                # Bottom edge at full bay height
                bottom_y = y2
                
                # Create polygon points in proper order (clockwise)
                polygon_coords_pixel = [
                    (x1, top_y),        # Top-left at 1/3 height
                    (x2, top_y),        # Top-right at 1/3 height  
                    (x2, bottom_y),     # Bottom-right at bay edge
                    (x1, bottom_y)      # Bottom-left at bay edge
                ]
                
                # Normalize coordinates to 0-1 range
                polygon_coords_normalized = [
                    (x / width, y / height) for x, y in polygon_coords_pixel
                ]
                
                regions.append(DetectedRegion(
                    region_type='polygon',
                    coordinates=polygon_coords_normalized,
                    confidence=space['confidence'],
                    suggested_alert='vehicle_infringement',
                    alert_config={
                        'detection_type': 'vehicle',
                        'action': 'parking_violation_detection',
                        'time_threshold': 300,
                        'confidence_threshold': 0.7
                    })
                )
            
            # 2. Detect other objects in parking lot (benches, signs, etc.)
            bench_objects = [det for det in detections if det['class_name'] in ['bench', 'chair', 'couch']]
            person_objects = [det for det in detections if det['class_name'] == 'person']
            other_objects = [det for det in detections if det['class_name'] not in 
                           ['car', 'truck', 'bus', 'person'] and det['confidence'] > 0.5]
            
            # Process bench/seating areas
            for obj in bench_objects:
                x1, y1, x2, y2 = obj['bbox']
                
                # Expand region around the bench for dwell time monitoring
                margin_x = max(30, (x2 - x1) * 0.5)
                margin_y = max(30, (y2 - y1) * 0.5)
                
                polygon_coords_pixel = [
                    (max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))),
                    (min(width, int(x2 + margin_x)), max(0, int(y1 - margin_y))),
                    (min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))),
                    (max(0, int(x1 - margin_x)), min(height, int(y2 + margin_y)))
                ]
                
                polygon_coords_normalized = [
                    (x / width, y / height) for x, y in polygon_coords_pixel
                ]
                
                regions.append(DetectedRegion(
                    region_type='polygon',
                    coordinates=polygon_coords_normalized,
                    confidence=obj['confidence'],
                    suggested_alert='dwell_time_detection',
                    alert_config={
                        'detection_type': 'person',
                        'action': 'monitor_dwell_time',
                        'time_threshold': 600,  # 10 minutes
                        'confidence_threshold': 0.6,
                        'description': f'Bench/seating area monitoring'
                    }
                ))

        elif scene_type in ['entrance', 'general_door']:
            # Get alert config for this scene type
            alert_config = self.alert_recommendations.get(scene_type, {})
            
            # Generate line across detected doors
            door_detections = [det for det in detections if det['class_name'] == 'door']
            
            if door_detections:
                for door in door_detections:
                    x1, y1, x2, y2 = door['bbox']
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    line_length = (x2 - x1) * 1.2  # Extend line beyond door width
                    
                    line_coords_pixel = [
                        (int(center_x - line_length // 2), center_y),
                        (int(center_x + line_length // 2), center_y)
                    ]
                    
                    # Normalize coordinates to 0-1 range
                    line_coords_normalized = [
                        (x / width, y / height) for x, y in line_coords_pixel
                    ]
                    
                    regions.append(DetectedRegion(
                        region_type='line',
                        coordinates=line_coords_normalized,
                        confidence=door['confidence'],
                        suggested_alert=alert_config.get('alert_type', 'line_crossing'),
                        alert_config=alert_config.get('config', {})
                    ))
                
        elif scene_type == 'office_desk':
            # Get alert config for this scene type
            alert_config = self.alert_recommendations.get(scene_type, {})
            
            # Generate polygon around desk/chair area
            chair_detections = [det for det in detections if det['class_name'] == 'chair']
            monitor_detections = [det for det in detections if det['class_name'] in ['tvmonitor', 'laptop']]
            
            # Combine chair and monitor detections to define workspace
            workspace_objects = chair_detections + monitor_detections
            
            if workspace_objects:
                # Calculate bounding box that encompasses all workspace objects
                min_x = min([obj['bbox'][0] for obj in workspace_objects])
                min_y = min([obj['bbox'][1] for obj in workspace_objects])
                max_x = max([obj['bbox'][2] for obj in workspace_objects])
                max_y = max([obj['bbox'][3] for obj in workspace_objects])
                
                # Expand the workspace area
                margin_x = (max_x - min_x) * 0.3
                margin_y = (max_y - min_y) * 0.3
                
                polygon_coords_pixel = [
                    (max(0, int(min_x - margin_x)), max(0, int(min_y - margin_y))),
                    (min(width, int(max_x + margin_x)), max(0, int(min_y - margin_y))),
                    (min(width, int(max_x + margin_x)), min(height, int(max_y + margin_y))),
                    (max(0, int(min_x - margin_x)), min(height, int(max_y + margin_y)))
                ]
                
                # Normalize coordinates to 0-1 range
                polygon_coords_normalized = [
                    (x / width, y / height) for x, y in polygon_coords_pixel
                ]
                
                regions.append(DetectedRegion(
                    region_type='polygon',
                    coordinates=polygon_coords_normalized,
                    confidence=max([obj['confidence'] for obj in workspace_objects]),
                    suggested_alert=alert_config.get('alert_type', 'region_intrusion'),
                    alert_config=alert_config.get('config', {})
                ))
        
        elif scene_type == 'warehouse':
            # Create safety monitoring zones
            alert_config = self.alert_recommendations.get(scene_type, {})
            
            # Divide warehouse into grid zones for safety monitoring
            grid_rows = 2
            grid_cols = 3
            cell_width = width // grid_cols
            cell_height = height // grid_rows
            
            for row in range(grid_rows):
                for col in range(grid_cols):
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = (col + 1) * cell_width
                    y2 = (row + 1) * cell_height
                    
                    polygon_coords_pixel = [
                        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                    ]
                    
                    polygon_coords_normalized = [
                        (x / width, y / height) for x, y in polygon_coords_pixel
                    ]
                    
                    regions.append(DetectedRegion(
                        region_type='polygon',
                        coordinates=polygon_coords_normalized,
                        confidence=0.8,
                        suggested_alert=alert_config.get('alert_type', 'safety_monitoring'),
                        alert_config=alert_config.get('config', {})
                    ))
        
        elif scene_type == 'retail':
            # Create customer analytics zones
            alert_config = self.alert_recommendations.get(scene_type, {})
            
            # Create entrance/exit lines if doors detected
            door_detections = [det for det in detections if det['class_name'] == 'door']
            
            for door in door_detections:
                x1, y1, x2, y2 = door['bbox']
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                line_length = (x2 - x1) * 1.5
                
                line_coords_pixel = [
                    (int(center_x - line_length // 2), center_y),
                    (int(center_x + line_length // 2), center_y)
                ]
                
                line_coords_normalized = [
                    (x / width, y / height) for x, y in line_coords_pixel
                ]
                
                regions.append(DetectedRegion(
                    region_type='line',
                    coordinates=line_coords_normalized,
                    confidence=door['confidence'],
                    suggested_alert='customer_counting',
                    alert_config={
                        'detection_type': 'person',
                        'action': 'count_in_out',
                        'direction': 'bidirectional',
                        'confidence_threshold': 0.6
                    }
                ))
            
            # Create dwell zones in retail space
            # Divide into zones for customer behavior analysis
            zone_width = width // 3
            zone_height = height // 2
            
            for i in range(3):
                x1 = i * zone_width
                x2 = (i + 1) * zone_width
                y1 = height // 3  # Start from middle of image
                y2 = height - 50  # Leave some margin at bottom
                
                polygon_coords_pixel = [
                    (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                ]
                
                polygon_coords_normalized = [
                    (x / width, y / height) for x, y in polygon_coords_pixel
                ]
                
                regions.append(DetectedRegion(
                    region_type='polygon',
                    coordinates=polygon_coords_normalized,
                    confidence=0.7,
                    suggested_alert='customer_dwell_time',
                    alert_config={
                        'detection_type': 'person',
                        'action': 'analyze_dwell_time',
                        'time_threshold': 120,  # 2 minutes
                        'confidence_threshold': 0.6,
                        'description': f'Retail zone {i+1} monitoring'
                    }
                ))
        
        return regions

    def analyze_scene(self, image_path: str) -> SceneAnalysisResult:
        """
        Complete scene analysis pipeline
        
        Args:
            image_path: Path to the input image
            
        Returns:
            SceneAnalysisResult containing all analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        print(f"Analyzing image: {width}x{height}")
        
        # Step 1: Detect objects
        print("Detecting objects...")
        detections = self.detect_objects(image_rgb)
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f" - {det['class_name']} (confidence: {det['confidence']:.2f})")
        
        # Step 2: Classify with Places365 (if enabled)
        places365_predictions = []
        if self.use_places365:
            print("\nClassifying with Places365...")
            places365_predictions = self.classify_with_places365(image_rgb, top_k=5)
            print("Top 5 Places365 predictions:")
            for scene, conf in places365_predictions:
                print(f" - {scene}: {conf:.3f}")
        
        # Step 3: Classify scene (combining YOLO and Places365)
        print("\nClassifying scene (combined)...")
        scene_type, scene_confidence = self.classify_scene(detections, (height, width), places365_predictions)
        print(f"Final Scene Classification: {scene_type} (confidence: {scene_confidence:.2f})")
        
        # Step 4: Generate regions
        print("\nGenerating suggested regions...")
        regions = self.generate_regions(scene_type, detections, image_rgb)
        print(f"Generated {len(regions)} region suggestions")
        
        return SceneAnalysisResult(
            scene_type=scene_type,
            confidence=scene_confidence,
            regions=regions,
            raw_detections=detections,
            places365_predictions=places365_predictions
        )
    
    def visualize_results(self, image_path: str, result: SceneAnalysisResult, 
                         save_path: str = None):
        """
        Visualize the analysis results on the image
        
        Args:
            image_path: Path to the original image
            result: SceneAnalysisResult to visualize
            save_path: Optional path to save the visualization
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.imshow(image_rgb)
        
        # Draw detected objects
        for detection in result.raw_detections:
            x1, y1, x2, y2 = detection['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='blue', 
                                   facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{detection['class_name']}: {detection['confidence']:.2f}", 
                   color='blue', fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Draw suggested regions with denormalized coordinates for visualization
        colors = ['red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
        region_labels = {}
        
        for i, region in enumerate(result.regions):
            color = colors[i % len(colors)]
            
            # Group similar alert types for better labeling
            if region.suggested_alert not in region_labels:
                region_labels[region.suggested_alert] = 0
            region_labels[region.suggested_alert] += 1
            
            label_text = f"{region.suggested_alert}"
            if region_labels[region.suggested_alert] > 1:
                label_text += f" #{region_labels[region.suggested_alert]}"
            
            if region.region_type == 'polygon':
                # Convert normalized coordinates back to pixel coordinates for visualization
                denormalized_coords = [(x * image_rgb.shape[1], y * image_rgb.shape[0]) 
                                     for x, y in region.coordinates]
                
                polygon = patches.Polygon(denormalized_coords, 
                                        linewidth=3, edgecolor=color, 
                                        facecolor=color, alpha=0.2)
                ax.add_patch(polygon)
                # Add label
                center_x = sum([p[0] for p in denormalized_coords]) / len(denormalized_coords)
                center_y = sum([p[1] for p in denormalized_coords]) / len(denormalized_coords)
                ax.text(center_x, center_y, label_text, 
                       color=color, fontsize=9, weight='bold', 
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            elif region.region_type == 'line':
                # Convert normalized coordinates back to pixel coordinates for visualization
                denormalized_coords = [(x * image_rgb.shape[1], y * image_rgb.shape[0]) 
                                     for x, y in region.coordinates]
                
                x_coords = [p[0] for p in denormalized_coords]
                y_coords = [p[1] for p in denormalized_coords]
                ax.plot(x_coords, y_coords, color=color, linewidth=4, alpha=0.8)
                # Add label
                mid_x = sum(x_coords) / len(x_coords)
                mid_y = sum(y_coords) / len(y_coords)
                ax.text(mid_x, mid_y-20, label_text, 
                       color=color, fontsize=9, weight='bold', 
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Add Places365 predictions to title if available
        title = f"Scene Analysis: {result.scene_type} (confidence: {result.confidence:.2f})"
        if result.places365_predictions:
            title += f"\nPlaces365 Top-3: "
            title += ", ".join([f"{s}({c:.2f})" for s, c in result.places365_predictions[:3]])
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


# Module-level function for exporting results
def export_results_to_json(result: SceneAnalysisResult, output_path: str):
    """
    Export analysis results to JSON format
    
    Args:
        result: SceneAnalysisResult to export
        output_path: Path to save JSON file
    """
    export_data = {
        'scene_analysis': {
            'scene_type': result.scene_type,
            'confidence': result.confidence,
            'places365_predictions': [
                {'scene': scene, 'confidence': conf} 
                for scene, conf in result.places365_predictions
            ],
            'suggested_regions': [
                {
                    'region_type': region.region_type,
                    'coordinates': region.coordinates,
                    'confidence': region.confidence,
                    'suggested_alert': region.suggested_alert,
                    'alert_config': region.alert_config
                }
                for region in result.regions
            ]
        },
        'raw_detections': result.raw_detections
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Results exported to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize the detector with Places365
    print("Initializing Scene Detector with Places365...")
    detector = SceneDetector(use_places365=True)
    
    # Example image path (replace with your actual image)
    image_path = "test_image.jpg"
    
    try:
        # Analyze the scene
        result = detector.analyze_scene(image_path)
        
        # Print results
        print("\n" + "="*50)
        print("SCENE ANALYSIS RESULTS")
        print("="*50)
        print(f"Scene Type: {result.scene_type}")
        print(f"Confidence: {result.confidence:.2f}")
        
        if result.places365_predictions:
            print(f"\nPlaces365 Top-5 Predictions:")
            for scene, conf in result.places365_predictions:
                print(f"  - {scene}: {conf:.3f}")
        
        print(f"\nSuggested Regions ({len(result.regions)}):")
        for i, region in enumerate(result.regions, 1):
            print(f"\nRegion {i}:")
            print(f"  Type: {region.region_type}")
            print(f"  Coordinates: {len(region.coordinates)} points")
            print(f"  Suggested Alert: {region.suggested_alert}")
            print(f"  Configuration: {region.alert_config}")
        
        # Visualize results
        detector.visualize_results(image_path, result, "analysis_result_places365.png")
        
        # Export to JSON
        export_results_to_json(result, "analysis_result_places365.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease make sure you have the required dependencies installed:")
        print("pip install ultralytics opencv-python matplotlib pillow torch torchvision")