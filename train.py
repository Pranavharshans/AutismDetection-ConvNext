import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from sklearn.metrics import accuracy_score
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
data_dir = "/teamspace/studios/this_studio/dataset"  # Replace this

# Enhanced transforms with more augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # increased variation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),                     # new
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(degrees=25),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.3),  # new
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validation transform without random augmentations
val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset with appropriate transforms
train_val_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

# Split sizes
train_size = int(0.7 * len(train_val_dataset))
val_size = int(0.15 * len(train_val_dataset))
test_size = len(train_val_dataset) - train_size - val_size

# Generate splits with fixed random seed for reproducibility
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(
    train_val_dataset, 
    [train_size, val_size, test_size],
    generator=generator
)

# Override transform for validation and test sets
val_set.dataset.transform = val_transform
test_set.dataset.transform = val_transform

# Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=64, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=64, num_workers=4, pin_memory=True)

# Load ConvNeXt
model = models.convnext_large(pretrained=True)

# Freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
num_classes = len(train_val_dataset.classes)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training function
def train(model, loader):
    model.train()
    running_loss = 0
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / len(loader), correct / total

# Validation function
def validate(model, loader):
    model.eval()
    running_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / len(loader), correct / total

# Training loop with learning rate scheduler
epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

best_val_acc = 0
best_model_path = "best_convnext_model.pth"

for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    # Step the scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    print(f"Learning Rate: {current_lr:.2e}\n")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with validation accuracy: {val_acc*100:.2f}%")

# Load best model before fine-tuning
model.load_state_dict(torch.load(best_model_path))
print(f"Loaded best model with validation accuracy: {best_val_acc*100:.2f}%")

# Unfreeze all for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Use a smaller learning rate for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
fine_tune_epochs = 20  # Increased from 5 to 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-7)

best_ft_val_acc = 0
best_ft_model_path = "best_convnext_finetuned.pth"

# Fine-tune
print("Fine-tuning entire model...")
for epoch in range(fine_tune_epochs):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    # Step the scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"[FT] Epoch {epoch+1}/{fine_tune_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    print(f"Learning Rate: {current_lr:.2e}\n")
    
    # Save best fine-tuned model
    if val_acc > best_ft_val_acc:
        best_ft_val_acc = val_acc
        torch.save(model.state_dict(), best_ft_model_path)
        print(f"Saved new best fine-tuned model with validation accuracy: {val_acc*100:.2f}%")

# Load best fine-tuned model before evaluation
model.load_state_dict(torch.load(best_ft_model_path))
print(f"Loaded best fine-tuned model with validation accuracy: {best_ft_val_acc*100:.2f}%")

# Evaluate on test set
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Test Accuracy: {acc*100:.2f}%")
    return acc

test_acc = evaluate(model, test_loader)

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_accuracy': test_acc,
    'classes': train_val_dataset.classes
}, "ASD_Detection-ConvNext.pth")

print("Training complete! Final model saved.")
