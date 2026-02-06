# Brain_Tumor_classification
This project implements a complete image classification pipeline for MRI-based brain tumor detection. The model classifies images into four categories:

- **glioma**
- **meningioma**
- **pituitary**
- **no_tumor**

The workflow includes training, saving the best model, and making predictions for individual images or full folders of MRI scans.

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

Ensure you have Python 3.8+ installed. Then install all required packages:

```bash
pip install -r requirements.txt
Note: If you wish to use GPU acceleration, ensure you have the appropriate version of PyTorch installed for your system (CUDA support).

Step 2: Setup Project Structure
Run the setup script to create all necessary directories:

Bash
python setup_project.py
This will create:

models/ - for storing trained model weights

Training/ - for training images

Testing/ - for validation images

PredictFolder/ - for images you want to classify

Step 3: Prepare Your Data
Organize your MRI images into the following structure:

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
    ├── patient_scan1.jpg
    └── ...
Important Notes:

Class folders must be named exactly: glioma, meningioma, pituitary, no_tumor

Supported image formats: .jpg, .jpeg, .png, .bmp

Step 4: Train the Model
Once your data is organized, train the model:

Bash
python train.py
Details:

Uses EfficientNet B3 architecture

Default: 10 epochs (modifiable in train.py)

Automatically saves the best model to models/best_effnet_b3.pth

Uses GPU if available, otherwise falls back to CPU

Step 5: Make Predictions
Option A: Command Line Prediction
Place MRI images inside the PredictFolder/ directory, then run:

Bash
python predict.py
The script will analyze all images and provide a final diagnosis summary based on the majority vote of tumor types found.

Option B: Streamlit Web Application
Launch the interactive web application:

Bash
streamlit run app.py
This interface provides:

Patient Overview: Diagnosis verdict and confidence

GradCAM Visualization: Visual explanations of where the model is looking

Detailed Analysis: Per-image probabilities

##Project Structure
Brain-Tumor-Detection-and-Classification/
├── app.py                  # Streamlit web application
├── model.py                # EfficientNet B3 model definition
├── train.py                # Training script
├── predict.py              # Command-line prediction script
├── gradcam.py              # GradCAM visualization utilities
├── viz_utils.py            # Visualization helper functions
├── setup_project.py        # Project setup script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── models/                 # Trained model weights
├── Training/               # Training dataset
├── Testing/                # Validation dataset
└── PredictFolder/          # Images for prediction
##Configuration
You can modify these parameters in the respective files:

train.py

Python
BATCH_SIZE = 16          # Adjust based on memory
LEARNING_RATE = 0.001    # Learning rate for optimizer
EPOCHS = 10              # Number of training epochs
IMG_SIZE = 300           # Image resize dimension
app.py

Python
IMG_SIZE = 300           # Image processing size
##System Requirements
Python: 3.8 or higher

RAM: 8GB minimum recommended

Compute: CPU is supported, but a GPU is recommended for faster training.

## Troubleshooting
"Model file not found": Run python train.py to generate the model file first.

"No images found": Add MRI images to the PredictFolder/ directory.

"CUDA out of memory": Reduce BATCH_SIZE in train.py (try 8 or 4).

Import Errors: Ensure all dependencies are installed via pip install -r requirements.txt.

## Model Details
Architecture: EfficientNet B3

Input: 300x300 RGB images

Output: 4 classes (glioma, meningioma, no_tumor, pituitary)

Loss Function: CrossEntropyLoss

Optimizer: Adam

##License
This project is for educational and research purposes.
