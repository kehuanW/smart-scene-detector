"""
Data models for the monitoring system
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from config_generator import generate_features_config

@dataclass
class MonitoringStandard:
    """Structure to hold monitoring standard information"""
    standard_id: int
    shape: str  # 'polygon' or 'line'
    features: List[str]
    coordinates: List[Tuple[float, float]]  # Normalized coordinates (0-1)
    confidence: float
    reasoning: str
    note: str = ""
    features_config: Dict[str, Any] = None

    def __post_init__(self):
        """Generate features_config based on recommended features"""
        if self.features_config is None:
            self.features_config = generate_features_config(self.features)

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