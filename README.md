#Brain Tumor Classification
This project implements an MRI image classification pipeline for detecting brain tumors.  
The model classifies images into four categories:

- glioma  
- meningioma  
- pituitary  
- no_tumor  

The workflow includes training, saving the best model, and predicting tumor type from images or folders.

---

## Quick Start

### 1. Install Dependencies

Ensure Python 3.8+ is installed, then run:

```bash
pip install -r requirements.txt
If PyTorch installation fails:

pip install torch torchvision
2. Setup Project Structure
python setup_project.py
This creates:

models/ – trained model weights

Training/ – training images

Testing/ – validation images

PredictFolder/ – images for prediction

3. Prepare Dataset
Project folder structure:

Brain-Tumor-Detection-and-Classification/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── PredictFolder/
Notes

Folder names must exactly match the class names above.

Supported formats: .jpg, .jpeg, .png, .bmp.

4. Train the Model
python train.py
Training configuration

Architecture: EfficientNet-B3

Epochs: 10

Batch size: 16

Best model saved to:

models/best_effnet_b3.pth
5. Make Predictions
Command-line prediction
python predict.py
The script analyzes all images in PredictFolder and outputs the most likely tumor type.

Streamlit Web App
streamlit run app.py
Provides:

Diagnosis summary

Image-wise predictions

GradCAM visual explanation

Project Structure
Brain-Tumor-Detection-and-Classification/
├── app.py
├── model.py
├── train.py
├── predict.py
├── gradcam.py
├── viz_utils.py
├── setup_project.py
├── requirements.txt
├── README.md
├── models/
├── Training/
├── Testing/
└── PredictFolder/
Configuration Parameters
train.py
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = 300
app.py
IMG_SIZE = 300
MAX_DETAIL_THUMBNAILS = 8
Minimum Requirements
Python 3.8+

8 GB RAM

CPU support for training and prediction

Troubleshooting
Model not found

python train.py
No images in prediction folder

Add images to PredictFolder/.

Torch not installed

pip install torch torchvision
Model Information
Architecture: EfficientNet-B3 (pretrained on ImageNet)

Input size: 300 × 300 RGB

Output: 4 tumor classes

Optimizer: Adam

Loss: CrossEntropyLoss

Workflow Summary
python setup_project.py
pip install -r requirements.txt
python train.py
python predict.py
License
For educational and research use only.
