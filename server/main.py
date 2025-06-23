#!/usr/bin/env python3
"""
Main entry point for the Computer Vision Monitoring System
"""

from monitoring_system import MonitoringStandardRecommender

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