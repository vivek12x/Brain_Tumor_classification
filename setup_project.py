# setup_project.py
"""
Setup script for Brain Tumor Classification Project
This script creates necessary directories and checks for required files.
"""
import os
from pathlib import Path

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent
    
    directories = [
        base_dir / "models",
        base_dir / "Training" / "glioma",
        base_dir / "Training" / "meningioma",
        base_dir / "Training" / "no_tumor",
        base_dir / "Training" / "pituitary",
        base_dir / "Testing" / "glioma",
        base_dir / "Testing" / "meningioma",
        base_dir / "Testing" / "no_tumor",
        base_dir / "Testing" / "pituitary",
        base_dir / "PredictFolder",
    ]
    
    print("Creating project directories...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✅ All directories created successfully!")
    
def check_model():
    """Check if the trained model exists."""
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "best_effnet_b3.pth"
    
    print("\nChecking for trained model...")
    if model_path.exists():
        print(f"✓ Model found at: {model_path}")
        return True
    else:
        print(f"⚠ Model NOT found at: {model_path}")
        print("  You need to train the model first by running: python train.py")
        return False

def check_data():
    """Check if training/testing data exists."""
    base_dir = Path(__file__).parent
    train_dir = base_dir / "Training"
    test_dir = base_dir / "Testing"
    
    print("\nChecking for training data...")
    
    train_images = sum(1 for _ in train_dir.rglob("*.jpg")) + \
                   sum(1 for _ in train_dir.rglob("*.png")) + \
                   sum(1 for _ in train_dir.rglob("*.jpeg"))
    
    test_images = sum(1 for _ in test_dir.rglob("*.jpg")) + \
                  sum(1 for _ in test_dir.rglob("*.png")) + \
                  sum(1 for _ in test_dir.rglob("*.jpeg"))
    
    if train_images > 0:
        print(f"✓ Found {train_images} training images")
    else:
        print("⚠ No training images found in Training/ directory")
        print("  Please add MRI images to the Training folder subfolders")
    
    if test_images > 0:
        print(f"✓ Found {test_images} testing images")
    else:
        print("⚠ No testing images found in Testing/ directory")
        print("  Please add MRI images to the Testing folder subfolders")
    
    return train_images > 0 and test_images > 0

def main():
    print("=" * 60)
    print("Brain Tumor Classification - Project Setup")
    print("=" * 60)
    
    setup_directories()
    has_model = check_model()
    has_data = check_data()
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("=" * 60)
    
    if has_data and has_model:
        print("✅ Project is ready to use!")
        print("\nYou can now:")
        print("  1. Run predictions: python predict.py")
        print("  2. Launch Streamlit app: streamlit run app.py")
    elif has_data and not has_model:
        print("⚠ Data found but model not trained")
        print("\nNext steps:")
        print("  1. Train the model: python train.py")
        print("  2. Then run predictions or launch the app")
    elif not has_data and has_model:
        print("⚠ Model found but no data")
        print("\nNext steps:")
        print("  1. Add training images to Training/ folder")
        print("  2. Add testing images to Testing/ folder")
        print("  3. Optionally retrain: python train.py")
    else:
        print("⚠ Setup incomplete")
        print("\nNext steps:")
        print("  1. Add training images to Training/ folder")
        print("  2. Add testing images to Testing/ folder")
        print("  3. Train the model: python train.py")
        print("  4. Run predictions or launch the app")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
