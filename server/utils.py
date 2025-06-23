"""
Utility functions for the monitoring system
"""

import os
import glob
from typing import List

def get_image_files(folder_path: str) -> List[str]:
    """Get all image files from a folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    return sorted(image_files)