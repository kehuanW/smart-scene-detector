import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import json

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

class SceneDetector:
    """Main scene detection and region suggestion system"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initialize the scene detector
        
        Args:
            model_path: Path to YOLO model (will download if not exists)
        """
        print("Loading YOLO model...")
        self.yolo_model = YOLO(model_path)
        
        # Scene classification rules based on detected objects
        self.scene_rules = {
            'car_park': {
                'required_objects': ['car'],
                'optional_objects': ['truck', 'bus'],
                'spatial_pattern': 'multiple_vehicles_arranged'
            },
            'entrance': {
                'required_objects': ['door'],
                'optional_objects': ['person'],
                'spatial_pattern': 'door_near_edge'
            },
            'office_desk': {
                'required_objects': ['chair'],
                'optional_objects': ['laptop', 'keyboard', 'mouse'],
                'spatial_pattern': 'desk_furniture_cluster'
            },
            'general_door': {
                'required_objects': ['door'],
                'optional_objects': [],
                'spatial_pattern': 'single_door'
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
            }
        }
    
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
    
    def classify_scene(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Tuple[str, float]:
        """
        Classify the scene based on detected objects and spatial patterns
        
        Args:
            detections: List of detected objects
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of (scene_type, confidence)
        """
        detected_classes = [det['class_name'] for det in detections]
        height, width = image_shape
        
        scene_scores = {}
        
        for scene_type, rules in self.scene_rules.items():
            score = 0
            
            # Check required objects
            required_found = sum(1 for obj in rules['required_objects'] if obj in detected_classes)
            if required_found == 0:
                scene_scores[scene_type] = 0
                continue
            
            score += required_found * 0.5
            
            # Check optional objects
            optional_found = sum(1 for obj in rules['optional_objects'] if obj in detected_classes)
            score += optional_found * 0.2
            
            # Spatial pattern analysis
            if scene_type == 'car_park':
                car_detections = [det for det in detections if det['class_name'] == 'car']
                if len(car_detections) >= 2:
                    score += 0.3  # Multiple cars suggest parking lot
                    
            elif scene_type == 'entrance' or scene_type == 'general_door':
                door_detections = [det for det in detections if det['class_name'] == 'door']
                for door in door_detections:
                    x1, y1, x2, y2 = door['bbox']
                    # Check if door is near image edge (typical for entrance)
                    if (x1 < width * 0.1 or x2 > width * 0.9 or 
                        y1 < height * 0.1 or y2 > height * 0.9):
                        if scene_type == 'entrance':
                            score += 0.3
                    else:
                        if scene_type == 'general_door':
                            score += 0.3
                            
            elif scene_type == 'office_desk':
                chair_detections = [det for det in detections if det['class_name'] == 'chair']
                if len(chair_detections) >= 1:
                    score += 0.3
            
            scene_scores[scene_type] = min(score, 1.0)
        
        # Return the scene with highest score
        if not scene_scores or max(scene_scores.values()) < 0.3:
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

        elif scene_type in ['entrance', 'general_door']:
            # Get alert config for this scene type
            alert_config = self.alert_recommendations.get(scene_type, {})
            
            # 2. Detect other objects in parking lot (benches, signs, etc.)
            # Expand object detection to include more object types
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
                
                # Normalize coordinates to 0-1 range
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
            
            # If no bench detected by YOLO, try to find bench-like objects in the background area
            if len(bench_objects) == 0:
                # Look for objects in the background/grass area (likely benches)
                background_objects = []
                for obj in detections:
                    x1, y1, x2, y2 = obj['bbox']
                    center_y = (y1 + y2) / 2
                    
                    # Objects in upper portion of image (background area)
                    if center_y < height * 0.4 and obj['class_name'] not in ['car', 'truck', 'person']:
                        background_objects.append(obj)
                
                # Create dwell time regions for background objects (likely benches/furniture)
                for obj in background_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # Create monitoring area around the object
                    margin_x = max(40, (x2 - x1) * 0.8)
                    margin_y = max(40, (y2 - y1) * 0.8)
                    
                    polygon_coords_pixel = [
                        (max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))),
                        (min(width, int(x2 + margin_x)), max(0, int(y1 - margin_y))),
                        (min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))),
                        (max(0, int(x1 - margin_x)), min(height, int(y2 + margin_y)))
                    ]
                    
                    # Normalize coordinates to 0-1 range
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
                            'time_threshold': 900,  # 15 minutes for general objects
                            'confidence_threshold': 0.6,
                            'description': f'Area monitoring around {obj["class_name"]}'
                        }
                    ))
            
            # Process other detected objects
            for obj in other_objects:
                x1, y1, x2, y2 = obj['bbox']
                
                alert_type = 'object_detection'
                alert_config = {
                    'detection_type': 'general_object',
                    'action': 'object_monitoring',
                    'confidence_threshold': 0.7,
                    'description': f'{obj["class_name"]} monitoring'
                }
                
                # Smaller expansion for general objects
                margin_x = max(15, (x2 - x1) * 0.2)
                margin_y = max(15, (y2 - y1) * 0.2)
                
                polygon_coords_pixel = [
                    (max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))),
                    (min(width, int(x2 + margin_x)), max(0, int(y1 - margin_y))),
                    (min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))),
                    (max(0, int(x1 - margin_x)), min(height, int(y2 + margin_y)))
                ]
                
                # Normalize coordinates to 0-1 range
                polygon_coords_normalized = [
                    (x / width, y / height) for x, y in polygon_coords_pixel
                ]
                
                regions.append(DetectedRegion(
                    region_type='polygon',
                    coordinates=polygon_coords_normalized,
                    confidence=obj['confidence'],
                    suggested_alert=alert_type,
                    alert_config=alert_config
                ))
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

            if chair_detections:
                for chair in chair_detections:
                    x1, y1, x2, y2 = chair['bbox']
                    # Create a larger polygon around the desk area
                    margin_x = (x2 - x1) * 0.5
                    margin_y = (y2 - y1) * 0.5
                    
                    polygon_coords_pixel = [
                        (max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))),
                        (min(width, int(x2 + margin_x)), max(0, int(y1 - margin_y))),
                        (min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))),
                        (max(0, int(x1 - margin_x)), min(height, int(y2 + margin_y)))
                    ]
                    
                    # Normalize coordinates to 0-1 range
                    polygon_coords_normalized = [
                        (x / width, y / height) for x, y in polygon_coords_pixel
                    ]
                    
                    regions.append(DetectedRegion(
                        region_type='polygon',
                        coordinates=polygon_coords_normalized,
                        confidence=chair['confidence'],
                        suggested_alert=alert_config.get('alert_type', 'region_intrusion'),
                        alert_config=alert_config.get('config', {})
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
        print(f"Found {len(detections)} objects, detected:")
        for det in detections:
            print(f" - {det['class_name']} (confidence: {det['confidence']:.2f})")

        # Step 2: Classify scene
        print("Classifying scene...")
        scene_type, scene_confidence = self.classify_scene(detections, (height, width))
        print(f"Scene: {scene_type} (confidence: {scene_confidence:.2f})")
        
        # Step 3: Generate regions
        print("Generating suggested regions...")
        regions = self.generate_regions(scene_type, detections, image_rgb)
        print(f"Generated {len(regions)} region suggestions")
        
        return SceneAnalysisResult(
            scene_type=scene_type,
            confidence=scene_confidence,
            regions=regions,
            raw_detections=detections
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
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Draw detected objects
        for detection in result.raw_detections:
            x1, y1, x2, y2 = detection['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='blue', 
                                   facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{detection['class_name']}: {detection['confidence']:.2f}", 
                   color='blue', fontsize=10, weight='bold')
        
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
        
        ax.set_title(f"Scene Analysis: {result.scene_type} (confidence: {result.confidence:.2f})", 
                    fontsize=14, weight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

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
    # Initialize the detector
    detector = SceneDetector()
    
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
        print(f"\nSuggested Regions ({len(result.regions)}):")
        
        for i, region in enumerate(result.regions, 1):
            print(f"\nRegion {i}:")
            print(f"  Type: {region.region_type}")
            print(f"  Coordinates: {region.coordinates}")
            print(f"  Suggested Alert: {region.suggested_alert}")
            print(f"  Configuration: {region.alert_config}")
        
        # Visualize results
        detector.visualize_results(image_path, result, "analysis_result.png")
        
        # Export to JSON
        export_results_to_json(result, "analysis_result.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure you have a test image and the required dependencies installed:")
        print("pip install ultralytics opencv-python matplotlib pillow")