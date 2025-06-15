#!/usr/bin/env python3
"""
Simple app.py for Scene Detector Demo
This file uses the SceneDetector class from scene_detector.py
"""

# Import the SceneDetector from your scene_detector.py file
from scene_detector import SceneDetector, export_results_to_json

def main():
    # Initialize the detector with Places365 enabled
    print("Initializing Scene Detector with Places365...")
    detector = SceneDetector(use_places365=True)
    
    # Specify your test image
    image_path = "test_image.jpg"  # Change this to your actual image path
    
    try:
        # Analyze the scene
        print(f"\nAnalyzing image: {image_path}")
        result = detector.analyze_scene(image_path)
        
        # Print results
        print("\n" + "="*50)
        print("SCENE ANALYSIS RESULTS")
        print("="*50)
        print(f"Scene Type: {result.scene_type}")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Show Places365 predictions if available
        if result.places365_predictions:
            print(f"\nPlaces365 Top-5 Predictions:")
            for scene, conf in result.places365_predictions:
                print(f"  - {scene}: {conf:.3f}")
        
        # Show detected objects
        print(f"\nDetected Objects ({len(result.raw_detections)}):")
        for det in result.raw_detections:
            print(f"  - {det['class_name']} (confidence: {det['confidence']:.2f})")
        
        # Show suggested regions
        print(f"\nSuggested Regions ({len(result.regions)}):")
        for i, region in enumerate(result.regions, 1):
            print(f"\nRegion {i}:")
            print(f"  Type: {region.region_type}")
            print(f"  Alert: {region.suggested_alert}")
        
        # Save visualization
        print("\nSaving visualization...")
        detector.visualize_results(image_path, result, "analysis_result.png")
        
        # Export to JSON
        print("Exporting results to JSON...")
        export_results_to_json(result, "analysis_result.json")
        
        print("\nAnalysis complete! Check analysis_result.png and analysis_result.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()