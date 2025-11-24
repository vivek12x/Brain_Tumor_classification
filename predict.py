import torch
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from model import get_model

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(r"models\best_effnet_b3.pth")
PREDICT_FOLDER = Path(r"PredictFolder")
IMG_SIZE = 300

# 1. DEFINE ALL CLASSES (Must match training order exactly for weight loading)
# Alphabetical order: 0=glioma, 1=meningioma, 2=no_tumor, 3=pituitary
ALL_CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# 2. CLASS TO REMOVE (The one we want to ignore)
IGNORED_CLASS = 'no_tumor'

def predict_patient_folder():
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Scanning folder: {PREDICT_FOLDER}")
    print(f"(!) EXCLUDING class: '{IGNORED_CLASS}' from results.\n")

    # --- Load Model (Must be 4 classes to match saved weights) ---
    model = get_model(num_classes=len(ALL_CLASS_NAMES), pretrained=False)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Error: Model file not found. Please run train.py first.")
        return
        
    model.to(DEVICE)
    model.eval()

    # --- Define Transform ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Initialize Counters for VALID classes only ---
    # We create a dictionary only for the classes that are NOT ignored
    results = {name: 0 for name in ALL_CLASS_NAMES if name != IGNORED_CLASS}
    total_images = 0

    # Find the index we need to suppress (e.g., index 2 for no_tumor)
    if IGNORED_CLASS in ALL_CLASS_NAMES:
        ignore_index = ALL_CLASS_NAMES.index(IGNORED_CLASS)
    else:
        ignore_index = -1 # Should not happen if names match

    # --- Iterate and Predict ---
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(PREDICT_FOLDER) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("No images found in the prediction folder.")
        return

    print(f"Processing {len(image_files)} images...")

    with torch.no_grad():
        for img_name in image_files:
            img_path = PREDICT_FOLDER / img_name
            
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)

                outputs = model(input_tensor)
                
                # --- THE FIX: Manually suppress the 'no_tumor' score ---
                # We set the value of the ignored index to Negative Infinity.
                # This ensures torch.max() NEVER picks it.
                if ignore_index != -1:
                    outputs[0, ignore_index] = -float('inf')

                # Determine winner among the remaining 3
                _, predicted_idx = torch.max(outputs, 1)
                predicted_class = ALL_CLASS_NAMES[predicted_idx.item()]
                
                # Double check we didn't pick the forbidden class (sanity check)
                if predicted_class != IGNORED_CLASS:
                    results[predicted_class] += 1
                    total_images += 1
                
            except Exception as e:
                print(f"Could not process {img_name}: {e}")

    # --- Final Summary ---
    print("\n" + "="*30)
    print("FINAL DIAGNOSIS SUMMARY (Tumor Types Only)")
    print("="*30)
    
    max_votes = -1
    winner = "Inconclusive"

    for class_name, count in results.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"{class_name.ljust(12)}: {count} images ({percentage:.1f}%)")
        
        if count > max_votes:
            max_votes = count
            winner = class_name

    print("-" * 30)
    print(f"VERDICT: The patient most likely has: {winner.upper()}")
    print("="*30)

if __name__ == "__main__":
    predict_patient_folder()