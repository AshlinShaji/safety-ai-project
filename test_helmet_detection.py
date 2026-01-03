import cv2
from ultralytics import YOLO
import os

print("=" * 60)
print("🎬 TESTING YOUR TRAINED HELMET DETECTION MODEL")
print("=" * 60)

# Find the best trained model
model_path = 'runs/detect/train6/weights/best.pt'

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("💡 Did training complete? Check runs/detect/ folder")
    exit()

print(f"\n📥 Loading your trained model from:")
print(f"   {model_path}")

try:
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

print("\n" + "=" * 60)
print("📷 STARTING WEBCAM")
print("=" * 60)
print("Controls:")
print("  Q - Quit")
print("  S - Save screenshot")
print("=" * 60 + "\n")

# Open camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera not found!")
    exit()

print("✅ Camera started!")
print("🎯 Point your camera at objects and watch for detection!")
print("💡 Try: yourself, a toy helmet, people, etc.\n")

frame_count = 0
saved_count = 0

while True:
    success, frame = camera.read()
    
    if not success:
        print("❌ Can't read from camera")
        break
    
    frame_count += 1
    
    # Only process every 2nd frame for speed
    if frame_count % 2 == 0:
        try:
            # Run detection with YOUR trained model
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            # Get all detections
            detections = results[0].boxes
            
            # Count helmets and people
            helmets = 0
            people = 0
            
            for detection in detections:
                class_id = int(detection.cls[0])
                class_name = results[0].names[class_id]
                confidence = float(detection.conf[0])
                
                if class_name == 'helmet':
                    helmets += 1
                elif class_name == 'person':
                    people += 1
            
            # Calculate safety status
            if people > 0:
                safety_percentage = (helmets / people) * 100
            else:
                safety_percentage = 0
            
            # Determine safety level
            if people == 0:
                status = "✅ SAFE - No people detected"
                color = (0, 255, 0)  # Green
            elif helmets == people:
                status = "✅ SAFE - All have helmets!"
                color = (0, 255, 0)  # Green
            elif safety_percentage >= 80:
                status = f"⚠️ WARNING - {safety_percentage:.0f}% have helmets"
                color = (0, 165, 255)  # Orange
            else:
                status = f"🚨 DANGER - Only {safety_percentage:.0f}% have helmets!"
                color = (0, 0, 255)  # Red
            
            # Add status text to frame
            cv2.putText(annotated_frame, status, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add counter
            cv2.putText(annotated_frame, 
                       f"Helmets: {helmets} | People: {people}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(annotated_frame, 
                       "Q=Quit | S=Save | Your Trained Model", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 1)
            
        except Exception as e:
            print(f"⚠️ Error during detection: {e}")
            annotated_frame = frame
    else:
        annotated_frame = frame
    
    # Show the frame
    cv2.imshow('🎯 Helmet Detection - YOUR TRAINED MODEL', annotated_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Quitting...")
        break
    
    if key == ord('s'):
        filename = f'helmet_detection_{frame_count}.png'
        cv2.imwrite(filename, annotated_frame)
        saved_count += 1
        print(f"📸 Saved: {filename}")

# Clean up
camera.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("✅ TEST COMPLETE!")
print("=" * 60)
print(f"📊 Statistics:")
print(f"   - Total frames processed: {frame_count}")
print(f"   - Screenshots saved: {saved_count}")
print(f"\n🎉 Your trained model is working!")
print("=" * 60)