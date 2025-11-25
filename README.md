# Brain Tumor Classification using EfficientNet B3

This project implements a complete image classification pipeline for MRI based brain tumor detection. The model classifies images into four categories:

- glioma
- meningioma
- pituitary
- no_tumor

The workflow includes training, saving the best model, and making predictions for individual images or full folders of MRI scans.

## Project Goal

Build a clean and reproducible pipeline that can:

1. Load and preprocess MRI images
2. Train an EfficientNet B3 model
3. Save the best model weights in the models folder
4. Predict using a single image or an entire folder
5. Produce a summary of how many images fall into each tumor category
6. Ignore no_tumor during final patient level diagnosis

## Directory Structure

Your project folder should look like this:

V3
└── Gemini
    ├── model.py
    ├── train.py
    ├── predict.py
    ├── models
    │   └── best_effnet_b3.pth
    ├── Training
    │   ├── glioma
    │   ├── meningioma
    │   ├── notumor
    │   └── pituitary
    ├── Testing
    │   ├── glioma
    │   ├── meningioma
    │   ├── notumor
    │   └── pituitary
    └── PredictFolder
        ├── image1.png
        ├── image2.jpg
        └── ...

## Requirements

Install dependencies:

pip install torch torchvision tqdm pillow

## Training the Model

Place images inside the Training and Testing directories.

Run:

python train.py

The script trains EfficientNet B3 and saves the best model to:

models/best_effnet_b3.pth

## Making Predictions

Place all MRI images of the patient inside the PredictFolder directory.

Run:

python predict.py

The script:

- Predicts each image
- Suppresses the no_tumor class
- Counts votes for glioma, meningioma, and pituitary
- Outputs final diagnosis by majority vote

Example:

glioma       : 12 images
meningioma   : 3 images
pituitary    : 1 image
VERDICT: GLIOMA

## Model Details

EfficientNet B3 is used with a custom classifier head:

Dropout(0.3)
Linear(1536 -> 4)

The model trains on GPU if available. Suitable for GPUs such as the RTX 4060.

## Notes

- Class folders must be named: glioma, meningioma, pituitary, no_tumor
- If your dataset uses notumor, rename it to no_tumor
- PredictFolder can contain mixed MRI sequences like FLAIR, T1, T1wCE, T2w
