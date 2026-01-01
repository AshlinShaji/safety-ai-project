import cv2
from ultralytics import YOLO

print("🤖 Loading YOLOv8 model...")
model = YOLO('yolov8s.pt')  # Load the small model

print("📷 Starting camera...")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera not found!")
    exit()

print("✅ Camera ready! Press Q to quit")
print("Watch for colored boxes around objects!")

frame_count = 0

while True:
    success, frame = camera.read()
    
    if not success:
        break
    
    frame_count += 1
    
    # Only process every 3rd frame (faster performance)
    if frame_count % 3 == 0:
        # Run YOLOv8 detection
        results = model(frame, verbose=False)
        
        # Draw boxes around detected objects
        annotated_frame = results[0].plot()
        
        # Add helpful text
        cv2.putText(annotated_frame, 
                   "Phase 1: YOLOv8 Detection - Press Q to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Print what was detected (in console)
        detections = results[0].boxes
        if len(detections) > 0 and frame_count % 30 == 0:  # Print every 30 frames
            print(f"\n🎯 Frame {frame_count} - Detected {len(detections)} object(s):")
            for i, detection in enumerate(detections):
                class_id = int(detection.cls[0])
                class_name = results[0].names[class_id]
                confidence = float(detection.conf[0]) * 100
                print(f"   - {class_name} ({confidence:.1f}% sure)")
    else:
        annotated_frame = frame
    
    # Show the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n✅ Detection complete!")
camera.release()
cv2.destroyAllWindows()