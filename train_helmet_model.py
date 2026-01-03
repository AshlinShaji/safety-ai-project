from ultralytics import YOLO
import os

print("=" * 60)
print("🚀 STARTING HELMET DETECTION TRAINING")
print("=" * 60)

# Check if dataset exists
dataset_yaml = 'datasets/helmet-detection/data.yaml'

if not os.path.exists(dataset_yaml):
    print(f"❌ ERROR: Dataset not found at {dataset_yaml}")
    print("❌ Did you download the dataset?")
    print("❌ Make sure folder structure is correct:")
    print("   safety-ai-project/")
    print("   └── datasets/")
    print("       └── helmet-detection/")
    print("           ├── images/")
    print("           ├── labels/")
    print("           └── data.yaml")
    exit()

print("✅ Dataset found!")
print(f"📂 Path: {dataset_yaml}")

# Fix the data.yaml file path issues
print("\n🔧 Fixing data.yaml configuration...")
import yaml

yaml_path = 'datasets/helmet-detection/data.yaml'
dataset_dir = os.path.abspath('datasets/helmet-detection')

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Update paths to use absolute paths
data['path'] = dataset_dir
data['train'] = os.path.join(dataset_dir, 'images/train')

# Check if 'valid' or 'val' folder exists
if os.path.exists(os.path.join(dataset_dir, 'images/valid')):
    data['val'] = os.path.join(dataset_dir, 'images/valid')
    print("   ℹ️ Found 'valid' folder (using it)")
else:
    data['val'] = os.path.join(dataset_dir, 'images/val')
    print("   ℹ️ Using 'val' folder")

# Handle test folder if it exists
if os.path.exists(os.path.join(dataset_dir, 'images/test')):
    data['test'] = os.path.join(dataset_dir, 'images/test')

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

print("✅ data.yaml fixed!")

print("\n" + "=" * 60)
print("📥 Loading YOLOv8 model...")
print("=" * 60)

# Load the small YOLOv8 model
model = YOLO('yolov8s.pt')

print("✅ Model loaded!")

print("\n" + "=" * 60)
print("🧠 STARTING TRAINING (this will take 15-30 minutes)")
print("=" * 60)
print("💡 DO NOT close this window!")
print("💡 Training will show progress below...")
print("=" * 60 + "\n")

try:
    # Train the model
    results = model.train(
        data=dataset_yaml,      # Path to your dataset
        epochs=50,              # Train for 50 rounds (complete passes)
        imgsz=416,              # Image size
        device='cpu',           # Use CPU (no GPU available)
        patience=10,            # Stop early if no improvement for 10 epochs
        batch=8,                # Reduced batch size for CPU
        save=True,              # Save the trained model
        verbose=True,           # Show detailed progress
        project='runs/detect',  # Where to save results
        name='train'            # Name of training session
    )
    
    print("\n" + "=" * 60)
    print("✅ ✅ ✅ TRAINING COMPLETE! ✅ ✅ ✅")
    print("=" * 60)
    
    print("\n📊 Training Results:")
    print(f"   📈 Final Loss: {results.results_dict}")
    
    print("\n🎉 Your trained model saved at:")
    print("   📁 runs/detect/train/weights/best.pt")
    
    print("\n📁 Training data saved at:")
    print("   📁 runs/detect/train/")
    print("      - results.csv (performance metrics)")
    print("      - confusion_matrix.png (accuracy chart)")
    print("      - plots/ (training graphs)")
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEP: Test your trained model!")
    print("   Run: python test_helmet_detection.py")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n❌ Training stopped by user")
    exit()
    
except Exception as e:
    print(f"\n❌ Training failed with error:")
    print(f"   {e}")
    print("\n💡 Troubleshooting tips:")
    print("   1. Make sure dataset is downloaded correctly")
    print("   2. Check you have at least 2GB free disk space")
    print("   3. Try reducing batch size: change batch=16 to batch=8")
    print("   4. Try using CPU instead: change device=0 to device='cpu'")
    exit()