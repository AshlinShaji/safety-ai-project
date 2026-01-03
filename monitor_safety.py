import cv2
from ultralytics import YOLO
import os
from safety_decision_engine import SafetyDecisionEngine, Detection

print("=" * 60)
print("🚨 REAL-TIME SAFETY MONITORING SYSTEM")
print("=" * 60)

# Load model
model_path = 'runs/detect/train6/weights/best.pt'
if not os.path.exists(model_path):
    print(f"❌ Model not found!")
    exit()

print("📥 Loading model...")
model = YOLO(model_path)

# Create decision engine
print("🧠 Initializing safety decision engine...")
engine = SafetyDecisionEngine()

print("\n📷 Starting safety monitoring...")
print("Controls: Q=Quit, S=Save screenshot, V=View violations\n")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera not found!")
    exit()

frame_count = 0
violation_count = 0

while True:
    success, frame = camera.read()
    
    if not success:
        break
    
    frame_count += 1
    
    # Process every 3rd frame for speed
    if frame_count % 3 == 0:
        # Run detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Convert detections to our format
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xywh[0].cpu().numpy()
            
            detections.append(Detection(
                object_type=class_name,
                confidence=confidence,
                bbox=tuple(bbox)
            ))
        
        # Make decision
        decision = engine.analyze_detections(detections, frame_count)
        
        # Get alert message and color
        alert_message = engine.get_alert_message(decision)
        alert_color = engine.get_alert_color(decision)
        
        # Update violation count
        if decision['safety_status'] == 'VIOLATION':
            violation_count += 1
        
        # Draw alerts on frame
        cv2.putText(annotated_frame, alert_message,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)
        
        cv2.putText(annotated_frame,
                   f"Helmets: {decision['helmets']} | People: {decision['people']} | Safety: {decision['safety_percentage']:.0f}%",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(annotated_frame,
                   f"Violations Detected: {violation_count}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(annotated_frame,
                   "Press V to see violations | Q=Quit",
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
    else:
        annotated_frame = frame
    
    cv2.imshow('🚨 Safety Monitoring System', annotated_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    if key == ord('v'):
        # Show violations
        stats = engine.get_statistics()
        print("\n" + "=" * 60)
        print("📊 VIOLATION STATISTICS")
        print("=" * 60)
        print(f"Total Violations: {stats['total_incidents']}")
        print(f"High Severity: {stats['high_severity']}")
        print(f"Medium Severity: {stats['medium_severity']}")
        print(f"Low Severity: {stats['low_severity']}")
        print("=" * 60 + "\n")
    
    if key == ord('s'):
        filename = f'safety_monitor_{frame_count}.png'
        cv2.imwrite(filename, annotated_frame)
        print(f"📸 Saved: {filename}")

# Clean up
camera.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("🛑 MONITORING STOPPED")
print("=" * 60)

# Show final statistics
stats = engine.get_statistics()
print(f"\n📊 FINAL STATISTICS:")
print(f"  Total Violations Detected: {stats['total_incidents']}")
print(f"  High Severity: {stats['high_severity']}")
print(f"  Medium Severity: {stats['medium_severity']}")
print(f"  Low Severity: {stats['low_severity']}")

# Save violations
engine.save_incidents('violations.json')

print("\n✅ Safety monitoring session complete!")
print("=" * 60)