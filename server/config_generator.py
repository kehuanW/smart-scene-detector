"""
Features configuration generator for monitoring standards
"""

from typing import List, Dict, Any

def generate_features_config(features: List[str]) -> Dict[str, Any]:
    """Generate features configuration based on recommended features"""
    
    # Base configuration structure with all features disabled
    config = {
        "COUNT": {
            "config_on": False,
            "features": {
                "standardCount": {
                    "config_on": False,
                    "objects": []
                },
                "objectsWithDwell": {
                    "config_on": False,
                    "rules": []
                },
                "people-occupancy": {
                    "config_on": False,
                    "number": 1
                }
            }
        },
        "DETECT": {
            "config_on": False,
            "features": {
                "objects": {
                    "config_on": False,
                    "objects": []
                }
            }
        },
        "LPR": {
            "config_on": False,
            "features": {
                "ANPR_DETECT": {
                    "config_on": False
                },
                "lpr-occupancy": {
                    "config_on": False,
                    "number": 1
                },
                "infringement": {
                    "config_on": False,
                    "ruleIds": [],
                    "whiteList": []
                },
                "blacklist": {
                    "config_on": False,
                    "plates": []
                }
            }
        },
        "REID": {
            "config_on": False,
            "features": {
                "retail": {
                    "config_on": False,
                    "type": "reid-journey"  # Default type, will be updated based on recommendation
                }
            }
        }
    }

    # Enable recommended features with preset rules and objects
    for feature in features:
        if feature == "standardCount":
            config["COUNT"]["config_on"] = True
            config["COUNT"]["features"]["standardCount"]["config_on"] = True
            # Add preset objects
            config["COUNT"]["features"]["standardCount"]["objects"] = [
                {"objName": "person"},
                {"objName": "dog"},
                {"objName": "car"},
                {"objName": "bicycle"},
                {"objName": "horse"}
            ]
        elif feature == "objectsWithDwell":
            config["COUNT"]["config_on"] = True
            config["COUNT"]["features"]["objectsWithDwell"]["config_on"] = True
            # Add preset rules
            config["COUNT"]["features"]["objectsWithDwell"]["rules"] = [
                {
                    "dwell": {"h": 0, "m": 1, "s": 0},
                    "id": 1,
                    "objName": "person",
                    "ruleName": "person-default"
                },
                {
                    "dwell": {"h": 0, "m": 1, "s": 0},
                    "id": 2,
                    "objName": "dog",
                    "ruleName": "dog-default"
                },
                {
                    "dwell": {"h": 0, "m": 1, "s": 0},
                    "id": 3,
                    "objName": "horse",
                    "ruleName": "horse-default"
                },
                {
                    "dwell": {"h": 0, "m": 1, "s": 0},
                    "id": 4,
                    "objName": "car",
                    "ruleName": "car-default"
                },
                {
                    "dwell": {"h": 0, "m": 1, "s": 0},
                    "id": 5,
                    "objName": "bicycle",
                    "ruleName": "bicycle-default"
                }
            ]
        elif feature == "people-occupancy":
            config["COUNT"]["config_on"] = True
            config["COUNT"]["features"]["people-occupancy"]["config_on"] = True
            # people-occupancy uses "number" field, not objects or rules
        elif feature == "objects":
            config["DETECT"]["config_on"] = True
            config["DETECT"]["features"]["objects"]["config_on"] = True
            # Add preset objects
            config["DETECT"]["features"]["objects"]["objects"] = [
                {"objName": "person"},
                {"objName": "dog"},
                {"objName": "horse"},
                {"objName": "car"},
                {"objName": "bicycle"}
            ]
        elif feature == "ANPR_DETECT":
            config["LPR"]["config_on"] = True
            config["LPR"]["features"]["ANPR_DETECT"]["config_on"] = True
            # ANPR_DETECT doesn't have objects or rules in the default config
        elif feature == "lpr-occupancy":
            config["LPR"]["config_on"] = True
            config["LPR"]["features"]["lpr-occupancy"]["config_on"] = True
            # lpr-occupancy uses "number" field, not objects or rules
        elif feature == "infringement":
            config["LPR"]["config_on"] = True
            config["LPR"]["features"]["infringement"]["config_on"] = True
            # infringement uses ruleIds and whiteList which are empty by default
        elif feature == "blacklist":
            config["LPR"]["config_on"] = True
            config["LPR"]["features"]["blacklist"]["config_on"] = True
            # blacklist uses plates which is empty by default
        elif feature == "reid-journey":
            config["REID"]["config_on"] = True
            config["REID"]["features"]["retail"]["config_on"] = True
            config["REID"]["features"]["retail"]["type"] = "reid-journey"
        elif feature == "reid-staff":
            config["REID"]["config_on"] = True
            config["REID"]["features"]["retail"]["config_on"] = True
            config["REID"]["features"]["retail"]["type"] = "reid-staff"

    return config