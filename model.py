import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=4, pretrained=True):
    """
    Loads EfficientNet-B3 and modifies the classifier head 
    for Brain Tumor Classification.
    """
    # Load EfficientNet-B3 weights
    weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    # Freeze feature extractor layers (optional: unfreeze for fine-tuning)
    # For a clean initial run, we often let them train or freeze the first blocks.
    # Here we will allow full training for best accuracy on MRI data.
    
    # Modify the final classification layer
    # EfficientNet-B3 classifier input features is 1536
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    return model

if __name__ == "__main__":
    # Test the model shape
    net = get_model()
    print("EfficientNet-B3 loaded successfully.")
    print(net.classifier)