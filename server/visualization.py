"""
Visualization functions for monitoring recommendations
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_models import MonitoringResult

def visualize_results(image_path: str, result: MonitoringResult, save_path: str):
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