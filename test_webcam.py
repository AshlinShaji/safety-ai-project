import cv2

# Turn on the camera (0 = your main webcam)
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    print("❌ Error: Camera not found!")
else:
    print("✅ Camera is ready!")

# Keep showing camera feed until you press 'q'
while True:
    # Read one frame from camera
    success, frame = camera.read()
    
    if not success:
        print("❌ Can't read from camera")
        break
    
    # Add text on the video
    cv2.putText(frame, "Phase 1: Webcam Test - Press Q to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Safety Monitor - Webcam Test', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("✅ Closing camera...")
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
print("✅ Done! Webcam test complete.")