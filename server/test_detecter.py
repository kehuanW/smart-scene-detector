#!/usr/bin/env python3
"""
Test script for Scene Detector with Places365
"""

import os
import sys

# Test if we can import the SceneDetector
try:
    from scene_detector import SceneDetector, SceneAnalysisResult, export_results_to_json
    print("✓ Successfully imported SceneDetector")
except ImportError as e:
    print(f"✗ Failed to import SceneDetector: {e}")
    print("Make sure the scene_detector.py file is in the same directory")
    sys.exit(1)

# Check for required dependencies
required_packages = [
    'cv2',
    'numpy',
    'torch',
    'torchvision',
    'ultralytics',
    'matplotlib',
    'PIL'
]

print("\nChecking dependencies:")
for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")
        print(f"  Install with: pip install {package}")

# Test initialization
print("\n" + "="*50)
print("Testing SceneDetector initialization...")
print("="*50)

try:
    detector = SceneDetector(use_places365=True)
    print("✓ SceneDetector initialized successfully")
    
    # Check if methods exist
    methods_to_check = [
        'analyze_scene',
        'detect_objects',
        'classify_scene',
        'classify_with_places365',
        'generate_regions',
        'visualize_results'
    ]
    
    print("\nChecking methods:")
    for method in methods_to_check:
        if hasattr(detector, method):
            print(f"✓ {method} method exists")
        else:
            print(f"✗ {method} method NOT FOUND")
            
except Exception as e:
    print(f"✗ Failed to initialize SceneDetector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with a sample image if provided
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"\n✗ Image not found: {image_path}")
        sys.exit(1)
        
    print(f"\n" + "="*50)
    print(f"Testing with image: {image_path}")
    print("="*50)
    
    try:
        result = detector.analyze_scene(image_path)
        print("✓ Analysis completed successfully!")
        
        print(f"\nResults:")
        print(f"- Scene Type: {result.scene_type}")
        print(f"- Confidence: {result.confidence:.2f}")
        print(f"- Objects Detected: {len(result.raw_detections)}")
        print(f"- Regions Suggested: {len(result.regions)}")
        
        if result.places365_predictions:
            print(f"\nPlaces365 Top Predictions:")
            for scene, conf in result.places365_predictions[:3]:
                print(f"  - {scene}: {conf:.3f}")
                
        # Save results
        output_json = "test_results.json"
        output_image = "test_visualization.png"
        
        export_results_to_json(result, output_json)
        print(f"\n✓ Results saved to {output_json}")
        
        detector.visualize_results(image_path, result, output_image)
        print(f"✓ Visualization saved to {output_image}")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n" + "="*50)
    print("USAGE:")
    print("="*50)
    print("To test with an image:")
    print(f"  python {sys.argv[0]} <path_to_image>")
    print("\nExample:")
    print(f"  python {sys.argv[0]} test_image.jpg")
    print("\nNote: Make sure scene_detector.py is in the same directory!")