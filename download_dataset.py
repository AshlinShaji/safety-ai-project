import os
import subprocess

print("📥 Downloading helmet detection dataset from GitHub...")

# Create datasets folder
os.makedirs('datasets/helmet-detection', exist_ok=True)

# Clone a helmet detection dataset from GitHub
try:
    print("⏳ Downloading... (this may take 1-3 minutes)")
    
    # This is a reliable GitHub dataset
    subprocess.run([
        'git', 'clone', 
        'https://github.com/evals-io/helmet-detection-yolov8.git',
        'datasets/helmet-detection'
    ], check=True)
    
    print("✅ Dataset downloaded successfully!")
    print("📁 Files are in: datasets/helmet-detection/")
    
except subprocess.CalledProcessError as e:
    print(f"⚠️ Git clone failed. Trying alternative method...")
    
    # Alternative: Download as ZIP from GitHub
    import urllib.request
    import zipfile
    
    url = "https://github.com/evals-io/helmet-detection-yolov8/archive/refs/heads/main.zip"
    filepath = 'datasets/helmet.zip'
    
    print("📥 Downloading ZIP file...")
    try:
        urllib.request.urlretrieve(url, filepath)
        
        print("📦 Extracting...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall('datasets/')
        
        # Rename folder
        os.rename('datasets/helmet-detection-yolov8-main', 'datasets/helmet-detection')
        os.remove(filepath)
        
        print("✅ Dataset ready!")
    except Exception as e2:
        print(f"❌ Download failed: {e2}")
        print("💡 Manual alternative below...")

except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Use OPTION C instead")