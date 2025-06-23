#!/usr/bin/env python3
"""
Enhanced Computer Vision Monitoring System
Integrates existing SceneDetector with monitoring standard recommendations
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torchvision.models as models
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import glob
from categories.places365_indoor_scenes import INDOOR_SCENES
from categories.places365_outdoor_scenes import OUTDOOR_SCENES
from categories.places365_parking_scenes import PARKING_SCENES

@dataclass
class MonitoringStandard:
    """Structure to hold monitoring standard information"""
    standard_id: int
    # type: str # "reid", "analytics", "lpr"
    shape: str  # 'polygon' or 'line'
    features: List[str]
    coordinates: List[Tuple[float, float]]  # Normalized coordinates (0-1)
    confidence: float
    reasoning: str
    note: str = ""

@dataclass
class MonitoringResult:
    """Complete monitoring analysis result"""
    image_path: str
    timestamp: str
    place365_results: List[Dict]
    yolo_detections: List[Dict]
    scene_type: str
    parking_detection: Dict = None
    recommendations: List[MonitoringStandard] = None

class MonitoringStandardRecommender:
    def __init__(self, model_path: str = 'yolov8n.pt', use_places365: bool = True):
        """
        Initialize the monitoring system
        
        Args:
            model_path: Path to YOLO model (supports custom trained models)
            use_places365: Whether to use Places365 for scene classification
        """
        print("🚀 Loading Computer Vision Monitoring System...")
        
        # Initialize YOLO model
        print(f"Loading YOLO model from: {model_path}")
        self.yolo_model = YOLO(model_path)
        self.confidence_threshold = 0.5
        
        # Initialize Places365 if requested
        self.use_places365 = use_places365
        self.place365_model = None
        self.place365_classes = None
        self.place365_transform = None
        
        if use_places365:
            self.place365_model, self.place365_classes = self.load_place365_model()
            self.place365_transform = self.get_place365_transform()
        
        # Feature definitions following our discussion
        self.features = [
            {'id': 'reid-journey', 'name': 'Journey', 'category': 'REID'},
            {'id': 'reid-staff', 'name': 'Staff', 'category': 'REID'},
            {'id': 'objects', 'name': 'Object Detect', 'category': 'DETECT'},
            {'id': 'standardCount', 'name': 'Object Count', 'category': 'COUNT'},
            {'id': 'people-occupancy', 'name': 'People Occupancy', 'category': 'COUNT'},
            {'id': 'objectsWithDwell', 'name': 'Dwell Time', 'category': 'COUNT'},
            {'id': 'ANPR_DETECT', 'name': 'LPR Detect', 'category': 'LPR'},
            {'id': 'blacklist', 'name': 'LPR Blacklist', 'category': 'LPR'},
            {'id': 'infringement', 'name': 'LPR Infringement', 'category': 'LPR'},
            {'id': 'lpr-occupancy', 'name': 'LPR Occupancy', 'category': 'LPR'}
        ]
        
        # Feature constraints
        self.none_lpr_features = ['reid-journey', 'reid-staff', 'standardCount', 'objects', 'people-occupancy', 'objectsWithDwell']
        self.lpr_features = ['ANPR_DETECT', 'blacklist', 'infringement', 'lpr-occupancy']
        self.line_compatible_features = ['reid-journey', 'objects', 'people-occupancy']
        self.people_only_features = ['reid-journey', 'reid-staff', 'people-occupancy']
        self.count_features = ['standardCount', 'objectsWithDwell', 'people-occupancy']
        
        # Scene classifications
        self.indoor_scenes = list(INDOOR_SCENES)
        self.outdoor_scenes = list(OUTDOOR_SCENES)
        self.car_park_scenes = list(PARKING_SCENES)

        print("✓ Monitoring System initialized successfully!")

    def load_place365_model(self):
        """Load Place365 ResNet50 model and class labels"""
        print("Loading Place365 model...")
        
        # Model architecture
        arch = 'resnet50'
        model = models.__dict__[arch](num_classes=365)
        
        # Download model weights if not present
        model_weight = f'{arch}_places365.pth.tar'
        if not os.path.exists(model_weight):
            weight_url = f'http://places2.csail.mit.edu/models_places365/{arch}_places365.pth.tar'
            print(f"Downloading Place365 model from {weight_url}")
            urllib.request.urlretrieve(weight_url, model_weight)
        
        # Load the pre-trained weights
        checkpoint = torch.load(model_weight, map_location=torch.device('cpu'))
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Download class labels if not present
        classes_file = 'categories_places365.txt'
        if not os.path.exists(classes_file):
            classes_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            print(f"Downloading Place365 classes from {classes_url}")
            urllib.request.urlretrieve(classes_url, classes_file)
        
        # Load class names
        classes = []
        with open(classes_file) as f:
            for line in f:
                classes.append(line.strip().split(' ')[0][3:])  # Remove '/a/', '/b/', etc.
        
        print(f"✓ Place365 model loaded with {len(classes)} classes")
        return model, classes

    def get_place365_transform(self):
        """Get the image transformation pipeline for Place365"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_place365(self, image_path):
        """Run Place365 prediction on image and return top 3 results"""
        if not self.use_places365 or self.place365_model is None:
            return self.get_mock_place365_results(image_path)
            
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            input_tensor = self.place365_transform(img).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                logit = self.place365_model(input_tensor)
                probs = F.softmax(logit, 1)
            
            # Get top 3 predictions
            top3_prob, top3_idx = torch.topk(probs, 3)
            
            results = []
            for i in range(3):
                idx = top3_idx[0][i].item()
                prob = top3_prob[0][i].item()
                scene_name = self.place365_classes[idx]
                results.append({"scene": scene_name, "confidence": round(prob, 3)})
            
            return results
            
        except Exception as e:
            print(f"Error in Place365 prediction: {e}")
            return self.get_mock_place365_results(image_path)

    def get_mock_place365_results(self, image_path):
        """Fallback mock results if Place365 fails"""
        filename = os.path.basename(image_path).lower()
        if 'parking' in filename or 'car' in filename:
            return [
                {'scene': 'parking_lot', 'confidence': 0.89},
                {'scene': 'gas_station', 'confidence': 0.08},
                {'scene': 'street', 'confidence': 0.03}
            ]
        elif 'office' in filename:
            return [
                {'scene': 'office', 'confidence': 0.92},
                {'scene': 'conference_room', 'confidence': 0.05},
                {'scene': 'lobby', 'confidence': 0.03}
            ]
        else:
            return [
                {'scene': 'pharmacy', 'confidence': 0.85},
                {'scene': 'drugstore', 'confidence': 0.12},
                {'scene': 'hospital_room', 'confidence': 0.03}
            ]

    def detect_objects_yolo(self, image_path):
        """Run YOLOv8 detection on image"""
        try:
            results = self.yolo_model(image_path, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        if confidence > self.confidence_threshold:
                            class_id = int(box.cls[0])
                            class_name = self.yolo_model.names[class_id]
                            
                            detections.append({
                                "object": class_name,
                                "confidence": round(confidence, 2)
                            })
            
            # Remove duplicates and keep highest confidence
            unique_detections = {}
            for det in detections:
                obj_name = det["object"]
                if obj_name not in unique_detections or det["confidence"] > unique_detections[obj_name]["confidence"]:
                    unique_detections[obj_name] = det
            
            return list(unique_detections.values())
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []

    def classify_indoor_outdoor(self, place365_results):
        # Same implementation as before
        top_scene = place365_results[0]["scene"]
        
        if '/indoor' in top_scene or top_scene.endswith('/indoor'):
            return "indoor"
        elif '/outdoor' in top_scene or top_scene.endswith('/outdoor'):
            return "outdoor"
            
        if top_scene in self.indoor_scenes:
            return "indoor"
        elif top_scene in self.outdoor_scenes:
            return "outdoor"
        
        # Fallback logic...
        indoor_confidence = sum(result.get("confidence", 0) 
                               for result in place365_results 
                               if result["scene"] in self.indoor_scenes)
        outdoor_confidence = sum(result.get("confidence", 0) 
                                for result in place365_results 
                                if result["scene"] in self.outdoor_scenes)
        
        return "indoor" if indoor_confidence >= outdoor_confidence else "outdoor"

    def detect_parking_spaces(self, image_path):
        """Enhanced parking space detection using computer vision"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # Method 1: Detect white lines (parking bay markings)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Clean up lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            parking_spaces = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Parking spaces are typically wider than tall
                    if 0.8 < aspect_ratio < 3.0 and w > 80 and h > 40:
                        # Create polygon coordinates (normalized)
                        polygon_coords = [
                            (x / width, y / height),
                            ((x + w) / width, y / height),
                            ((x + w) / width, (y + h) / height),
                            (x / width, (y + h) / height)
                        ]
                        
                        parking_spaces.append({
                            'space_id': f'P{len(parking_spaces) + 1}',
                            'polygon_coordinates': polygon_coords,
                            'status': 'detected',
                            'confidence': 0.9
                        })
            
            # If we don't find enough spaces, generate a grid
            if len(parking_spaces) < 4:
                parking_spaces = self.generate_parking_grid(width, height)
            
            return parking_spaces[:10]  # Limit to 10 spaces
            
        except Exception as e:
            print(f"Error in parking detection: {e}")
            return self.generate_parking_grid(800, 600)  # Default size

    def generate_parking_grid(self, width, height):
        """Generate a grid of parking spaces as fallback"""
        parking_spaces = []
        
        # Define parking area
        start_y = 0.4
        end_y = 0.75
        start_x = 0.1
        end_x = 0.9
        
        num_spaces = 6
        space_width = (end_x - start_x) / num_spaces
        
        for i in range(num_spaces):
            x1 = start_x + i * space_width
            x2 = start_x + (i + 1) * space_width - 0.01  # Small gap
            
            parking_spaces.append({
                'space_id': f'P{i + 1}',
                'polygon_coordinates': [
                    (x1, start_y),
                    (x2, start_y), 
                    (x2, end_y),
                    (x1, end_y)
                ],
                'status': 'generated',
                'confidence': 0.6
            })
        
        return parking_spaces

    def generate_recommendations(self, place365_results, yolo_detections, scene_type):
        """Generate monitoring standard recommendations based on scene analysis"""
        recommendations = []
        top_scene = place365_results[0]["scene"]
        detected_objects = [det["object"] for det in yolo_detections]
        
        # Initialize standard_id counter
        current_id = 1
        
        # Check what we have in the scene
        # people_detected = "person" in detected_objects
        # vehicles_detected = any(obj in detected_objects for obj in ["car", "truck", "bus", "motorcycle"])
        
        if scene_type == "indoor":
            # Rule: Journey for customer tracking at entrance/exit
            recommendations.append(MonitoringStandard(
                standard_id=current_id,
                shape="line",
                features=["reid-journey", "people-occupancy"],
                coordinates=[(0.1, 0.5), (0.9, 0.5)],  # Horizontal line across entrance
                confidence=0.85,
                reasoning=f"Indoor {top_scene} - track journey at entrance/exit"
            ))
            current_id += 1

            # Rule: Staff tracking (mutually exclusive with Journey per R4)
            recommendations.append(MonitoringStandard(
                standard_id=current_id,
                shape="polygon",
                features=["reid-staff"],
                coordinates=[(0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)],
                confidence=0.88,
                reasoning="Office environment - track staff presence and movement"
            ))
            current_id += 1
                
            # Rule: People occupancy for density monitoring
            recommendations.append(MonitoringStandard(
                standard_id=current_id,
                shape="polygon",
                features=["people-occupancy", "standardCount", "objects", "objectsWithDwell"],
                coordinates=[(0.2, 0.4), (0.6, 0.4), (0.6, 0.7), (0.2, 0.7)],
                confidence=0.82,
                reasoning=f"Monitor customer density in {top_scene} area"
            ))
            current_id += 1
    
        else:  # outdoor
            if top_scene in self.car_park_scenes:
                # Car park specific recommendations (following R1 - LPR features)
                # Rule: Infringement detection for parking violations
                recommendations.append(MonitoringStandard(
                    standard_id=current_id,
                    shape="polygon",
                    features=["infringement"],
                    coordinates=[(0.5, 0.1), (0.7, 0.1), (0.7, 0.3), (0.5, 0.3)],
                    confidence=0.92,
                    reasoning="Car park detected - monitor parking violations",
                    note="Will integrate with parking space detection when model is ready"
                ))
                current_id += 1
                
            # Rule: LPR occupancy for vehicle tracking
            recommendations.append(MonitoringStandard(
                standard_id=current_id,
                shape="polygon", 
                features=["lpr-occupancy", "LPR_DETECT", "blacklist"],
                coordinates=[(0.05, 0.35), (0.95, 0.35), (0.95, 0.8), (0.05, 0.8)],
                confidence=0.89,
                reasoning="Track vehicle occupancy in parking areas"
            ))
            current_id += 1

            # Rule: count/detect
            recommendations.append(MonitoringStandard(
                standard_id=current_id,
                shape="polygon",
                features=["people-occupancy", "standardCount", "objects", "objectsWithDwell"],
                coordinates=[(0.2, 0.3), (0.8, 0.3), (0.8, 0.7), (0.2, 0.7)],
                confidence=0.82,
                reasoning=f"Monitor customer density in {top_scene} area"
            ))
            current_id += 1
            
        # TODO: Validate recommendations against our rules
        # recommendations = self.validate_recommendations(recommendations)
        
        return recommendations

    def get_parking_detection_placeholder(self, parking_spaces=None):
        """Get parking detection structure"""
        if parking_spaces:
            return {
                "model_available": True,
                "detected_spaces": parking_spaces,
                "total_spaces": len(parking_spaces)
            }
        else:
            return {
                "model_available": False,
                "placeholder": "TODO: Implement parking space detection model",
                "expected_output": [
                    {
                        "space_id": "P1",
                        "polygon_coordinates": [[0.1, 0.4], [0.25, 0.4], [0.25, 0.7], [0.1, 0.7]],
                        "status": "occupied|vacant"
                    }
                ]
            }

    def process_image(self, image_path):
        """Process a single image and generate complete analysis"""
        print(f"\n📸 Processing: {os.path.basename(image_path)}")
        
        # Run Place365 prediction
        place365_results = self.predict_place365(image_path)
        
        # Run YOLO detection
        yolo_detections = self.detect_objects_yolo(image_path)
        
        # Classify indoor/outdoor
        scene_type = self.classify_indoor_outdoor(place365_results)
        
        # Generate parking detection if needed
        parking_detection = None
        if place365_results[0]["scene"] in self.car_park_scenes:
            parking_spaces = self.detect_parking_spaces(image_path)
            parking_detection = self.get_parking_detection_placeholder(parking_spaces)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(place365_results, yolo_detections, scene_type)
        
        # Convert recommendations to dict format
        recommendations_dict = []
        for rec in recommendations:
            recommendations_dict.append({
                "standard_id": rec.standard_id,
                "shape": rec.shape,
                "features": rec.features,
                "coordinates": rec.coordinates,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "note": rec.note
            })
        
        # Compile results
        result = MonitoringResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            place365_results=place365_results,
            yolo_detections=yolo_detections,
            scene_type=scene_type,
            parking_detection=parking_detection,
            recommendations=recommendations_dict
        )
        
        return result

    def visualize_results(self, image_path, result, save_path):
        """Visualize the monitoring recommendations on the image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.imshow(image_rgb)
            
            # Draw monitoring standards
            colors = ['red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
            
            for i, rec in enumerate(result.recommendations):
                color = colors[i % len(colors)]
                
                if rec["shape"] == "polygon":
                    # Convert normalized coordinates back to pixel coordinates
                    pixel_coords = [(x * width, y * height) for x, y in rec["coordinates"]]
                    polygon = patches.Polygon(pixel_coords, 
                                            linewidth=3, edgecolor=color, 
                                            facecolor=color, alpha=0.2)
                    ax.add_patch(polygon)
                    
                    # Find left upper corner of polygon
                    min_x = min([p[0] for p in pixel_coords])
                    min_y = min([p[1] for p in pixel_coords])
                    
                    features_text = "+".join(rec["features"])
                    ax.text(min_x + 5, min_y + 15, f"S{rec['standard_id']}: {features_text}", 
                        color=color, fontsize=9, weight='bold', 
                        ha='left', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
                
                elif rec["shape"] == "line":
                    # Convert normalized coordinates back to pixel coordinates
                    pixel_coords = [(x * width, y * height) for x, y in rec["coordinates"]]
                    x_coords = [p[0] for p in pixel_coords]
                    y_coords = [p[1] for p in pixel_coords]
                    ax.plot(x_coords, y_coords, color=color, linewidth=4, alpha=0.8)
                    
                    # Find left upper corner of line
                    min_x = min(x_coords)
                    min_y = min(y_coords)
                    
                    features_text = "+".join(rec["features"])
                    ax.text(min_x + 5, min_y - 5, f"S{rec['standard_id']}: {features_text}", 
                        color=color, fontsize=9, weight='bold', 
                        ha='left', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            # Add title with scene information
            title = f"Monitoring Standards: {result.scene_type}"
            if result.place365_results:
                title += f"\nTop Scene: {result.place365_results[0]['scene']} ({result.place365_results[0]['confidence']:.2f})"
            title += f"\nObjects: {', '.join([d['object'] for d in result.yolo_detections])}"
            
            ax.set_title(title, fontsize=12, weight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved: {save_path}")
            
        except Exception as e:
            print(f"Error in visualization: {e}")

    def process_folder(self, input_folder="images", output_folder="results"):
        """Process all images in input folder and save results"""
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_folder).glob(ext))
            image_files.extend(Path(input_folder).glob(ext.upper()))
        
        if not image_files:
            print(f"❌ No images found in {input_folder}")
            return
        
        print(f"📁 Found {len(image_files)} images to process")
        
        all_results = []
        
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"\n{'='*60}")
                print(f"Processing {i}/{len(image_files)}: {image_path.name}")
                print(f"{'='*60}")
                
                result = self.process_image(str(image_path))
                all_results.append(result)
                
                # Print summary
                print(f"Scene: {result.scene_type}")
                print(f"Objects: {len(result.yolo_detections)}")
                print(f"Standards: {len(result.recommendations)}")
                
                # Generate output filenames
                base_name = image_path.stem
                json_output = Path(output_folder) / f"{base_name}_monitoring.json"
                viz_output = Path(output_folder) / f"{base_name}_visualization.png"
                
                # Save individual result as JSON
                with open(json_output, 'w') as f:
                    # Convert result to dict for JSON serialization
                    result_dict = {
                        "image_path": result.image_path,
                        "timestamp": result.timestamp,
                        "place365_results": result.place365_results,
                        "yolo_detections": result.yolo_detections,
                        "scene_type": result.scene_type,
                        "recommendations": result.recommendations
                    }
                    
                    if result.parking_detection:
                        result_dict["parking_detection"] = result.parking_detection
                    
                    json.dump(result_dict, f, indent=2)
                
                # Create visualization
                self.visualize_results(str(image_path), result, str(viz_output))
                
                print(f"✓ Saved: {json_output.name}, {viz_output.name}")
                
            except Exception as e:
                print(f"✗ Error processing {image_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save combined summary
        summary_path = Path(output_folder) / "monitoring_summary.json"
        summary_data = {
            "total_images": len(image_files),
            "processed_successfully": len(all_results),
            "processing_timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "image_path": r.image_path,
                    "scene_type": r.scene_type,
                    "standards_count": len(r.recommendations),
                    "objects_detected": len(r.yolo_detections),
                    "top_scene": r.place365_results[0]["scene"] if r.place365_results else "unknown"
                }
                for r in all_results
            ]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n🎉 Processing complete!")
        print(f"✓ Processed: {len(all_results)}/{len(image_files)} images")
        print(f"✓ Results saved in: {output_folder}")
        print(f"✓ Summary: {summary_path.name}")

def get_image_files(folder_path):
    """Get all image files from a folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    return sorted(image_files)

def main():
    """Main function to run the monitoring system"""
    print("🚀 Computer Vision Monitoring System")
    print("=" * 50)
    
    try:
        # Configuration
        images_folder = "images"
        output_folder = "results"
        
        # You can specify your custom YOLO model path here
        # model_path = '/aria/personal/ParkingResults/saved_models/weights/best.pt'
        model_path = 'yolov8n.pt'  # Default model
        
        # Initialize the system
        recommender = MonitoringStandardRecommender(
            model_path=model_path,
            use_places365=True
        )
        
        # Process all images
        recommender.process_folder(
            input_folder=images_folder,
            output_folder=output_folder
        )
        
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📋 Setup Requirements:")
        print("1. Install: pip install ultralytics opencv-python torch torchvision pillow matplotlib")
        print("2. Create 'images' folder and add your images")
        print("3. Update model_path if using custom YOLO model")
        print("4. Run the script again")

if __name__ == "__main__":
    main()