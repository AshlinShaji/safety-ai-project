from dataclasses import dataclass
from datetime import datetime
from typing import List
import json

@dataclass
class Detection:
    """Represents a single detection"""
    object_type: str  # 'helmet' or 'person'
    confidence: float  # 0.0 to 1.0
    bbox: tuple  # (x, y, width, height)

@dataclass
class SafetyIncident:
    """Represents a safety violation"""
    timestamp: str
    incident_type: str  # 'no_helmet', 'no_vest', etc.
    severity: str  # 'low', 'medium', 'high'
    description: str
    frame_number: int

class SafetyDecisionEngine:
    """
    Makes decisions about safety violations
    Like a smart safety supervisor!
    """
    
    def __init__(self):
        self.incidents = []  # Track all violations
        self.safety_rules = {
            'require_helmet': True,
            'require_vest': False,  # Not checking for vest yet
            'min_detection_confidence': 0.5,  # 50% confidence minimum
        }
    
    def analyze_detections(self, detections: List[Detection], frame_num: int) -> dict:
        """
        Analyze detections and make decisions
        Returns: decision report
        """
        
        # Count what we found
        people_count = sum(1 for d in detections if d.object_type == 'person')
        helmets_count = sum(1 for d in detections if d.object_type == 'helmet')
        
        # Make decision
        decision = {
            'frame': frame_num,
            'timestamp': datetime.now().isoformat(),
            'people': people_count,
            'helmets': helmets_count,
            'violations': [],
            'safety_status': 'SAFE'
        }
        
        # Rule 1: Check for people without helmets
        if self.safety_rules['require_helmet']:
            if people_count > helmets_count:
                violation_count = people_count - helmets_count
                
                incident = SafetyIncident(
                    timestamp=datetime.now().isoformat(),
                    incident_type='no_helmet',
                    severity='high' if violation_count > 2 else 'medium',
                    description=f"{violation_count} person/people without helmet",
                    frame_number=frame_num
                )
                
                self.incidents.append(incident)
                decision['violations'].append(incident.__dict__)
                decision['safety_status'] = 'VIOLATION'
        
        # Calculate safety percentage
        if people_count > 0:
            safety_percentage = (helmets_count / people_count) * 100
        else:
            safety_percentage = 100  # No people = safe!
        
        decision['safety_percentage'] = safety_percentage
        
        return decision
    
    def get_alert_message(self, decision: dict) -> str:
        """
        Create a human-readable alert message
        """
        
        if decision['safety_status'] == 'SAFE':
            if decision['people'] == 0:
                return "✅ SAFE: No people detected"
            else:
                return f"✅ SAFE: All {decision['people']} person/people have helmets"
        
        else:  # VIOLATION
            people = decision['people']
            helmets = decision['helmets']
            missing = people - helmets
            
            return f"🚨 DANGER: {missing} person/people without helmet! ({helmets}/{people} safe)"
    
    def get_alert_color(self, decision: dict) -> tuple:
        """
        Return RGB color for alert
        Green = Safe, Red = Danger
        """
        
        safety_pct = decision.get('safety_percentage', 0)
        
        if safety_pct == 100:
            return (0, 255, 0)  # Green
        elif safety_pct >= 80:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red
    
    def save_incidents(self, filename='violations.json'):
        """
        Save all violations to a file
        """
        
        incident_data = [incident.__dict__ for incident in self.incidents]
        
        with open(filename, 'w') as f:
            json.dump(incident_data, f, indent=2)
        
        print(f"📄 Saved {len(self.incidents)} incidents to {filename}")
    
    def get_statistics(self) -> dict:
        """
        Get statistics about violations
        """
        
        return {
            'total_incidents': len(self.incidents),
            'high_severity': sum(1 for i in self.incidents if i.severity == 'high'),
            'medium_severity': sum(1 for i in self.incidents if i.severity == 'medium'),
            'low_severity': sum(1 for i in self.incidents if i.severity == 'low'),
        }


# Test the engine
if __name__ == "__main__":
    print("Testing Safety Decision Engine...\n")
    
    engine = SafetyDecisionEngine()
    
    # Simulate some detections
    test_detections_1 = [
        Detection('person', 0.95, (100, 50, 50, 100)),
        Detection('helmet', 0.92, (110, 40, 30, 25)),
    ]
    
    decision_1 = engine.analyze_detections(test_detections_1, frame_num=1)
    print("Test 1 (Safe):")
    print(f"  Alert: {engine.get_alert_message(decision_1)}")
    print(f"  Status: {decision_1['safety_status']}\n")
    
    # Simulate violation
    test_detections_2 = [
        Detection('person', 0.95, (100, 50, 50, 100)),
        Detection('person', 0.93, (200, 60, 50, 100)),
        Detection('helmet', 0.92, (110, 40, 30, 25)),
    ]
    
    decision_2 = engine.analyze_detections(test_detections_2, frame_num=50)
    print("Test 2 (Violation):")
    print(f"  Alert: {engine.get_alert_message(decision_2)}")
    print(f"  Status: {decision_2['safety_status']}\n")
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"Statistics: {stats}")