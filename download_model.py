from ultralytics import YOLO

print("📥 Downloading YOLOv8 model (this may take a minute)...")

# Download the small model (good balance of speed and accuracy)
model = YOLO('yolov8s.pt')

print("✅ Model downloaded successfully!")
print(f"Model saved to: {model.model_name}")