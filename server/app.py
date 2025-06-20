#!/usr/bin/env python3
"""
Simple app.py for Scene Detector Demo
This file uses the SceneDetector class from scene_detector.py
"""

import os
import glob
from pathlib import Path

# Import the SceneDetector from your scene_detector.py file
from scene_detector import SceneDetector, export_results_to_json

def get_image_files(folder_path):
    """Get all image files from a folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    return sorted(image_files)

def main():
    # Initialize the detector with Places365 enabled
    print("Initializing Scene Detector with Places365...")
    # I want to use record the absolute path of the my best.pth file, the path is 
    model_path = '/aria/personal/ParkingResults/saved_models/weights/best.pt'
    detector = SceneDetector(use_places365=True, model_path=model_path)
    
    # Specify your images folder
    images_folder = "images"  # Change this to your actual images folder path
    output_folder = "results"  # Output folder for results
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Get all image files
        image_files = get_image_files(images_folder)
        
        if not image_files:
            print(f"No image files found in {images_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            image_name = os.path.basename(image_path)
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(image_files)}: {image_name}")
            print(f"{'='*60}")
            
            try:
                # Analyze the scene
                result = detector.analyze_scene(image_path)
                
                # Print results
                print(f"Scene Type: {result.scene_type}")
                print(f"Confidence: {result.confidence:.2f}")
                
                # Show Places365 predictions if available
                if result.places365_predictions:
                    print(f"Places365 Top-3 Predictions:")
                    for scene, conf in result.places365_predictions[:3]:
                        print(f"  - {scene}: {conf:.3f}")
                
                # Show detected objects count
                print(f"Detected Objects: {len(result.raw_detections)}")
                
                # Generate output filenames
                base_name = os.path.splitext(image_name)[0]
                viz_output = os.path.join(output_folder, f"{base_name}_analysis.png")
                json_output = os.path.join(output_folder, f"{base_name}_analysis.json")
                
                # Save visualization
                detector.visualize_results(image_path, result, viz_output)
                
                # Export to JSON
                export_results_to_json(result, json_output)
                
                print(f"✓ Results saved: {viz_output}, {json_output}")
                
            except Exception as e:
                print(f"✗ Error processing {image_name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"Processed {len(image_files)} images")
        print(f"Results saved in: {output_folder}/")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()