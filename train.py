import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm # Progress bar
from model import get_model

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # Good fit for GTX 4060 (8GB VRAM) with EffNetB3
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = 300   # EfficientNet-B3 native resolution
SAVE_DIR = Path(r"models")

# --- Dataset Paths (Windows Safe) ---
TRAIN_DIR = Path(r"Training")
TEST_DIR = Path(r"Testing")

def get_transforms():
    """
    1. Resizes images to 300x300
    2. Converts to Tensor
    3. Normalizes using ImageNet stats (standard for EffNet)
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(), # Data augmentation
            transforms.RandomRotation(10),     # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_description(f"Training")
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total

def main():
    # Create models directory if not exists
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using Device: {DEVICE}")
    
    # 1. Prepare Data
    transforms_dict = get_transforms()
    
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transforms_dict['train'])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transforms_dict['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes found: {train_dataset.classes}")

    # 2. Setup Model
    model = get_model(num_classes=len(train_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler to reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    best_acc = 0.0

    # 3. Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, test_loader, criterion)
        
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = SAVE_DIR / "best_effnet_b3.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved to {save_path}")

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()