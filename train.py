import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import json

# =========================
# DEVICE (GPU SUPPORT 🔥)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

import torch
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor
# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),

    # 🔥 ADD HERE
    AddGaussianNoise(0., 0.05, p=0.3),       #noice ----injection

    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

batch_size = 32

# 🚨 REQUIRED FOR WINDOWS
def main():

    # =========================
    # DATASET
    # =========================
    train_data = datasets.ImageFolder("data/train", transform=train_transform)
    val_data   = datasets.ImageFolder("data/val", transform=val_transform)
    # 🔥 ADD THIS HERE
    print("Classes:", train_data.classes)  #which class is ai and real

    # 🔥 FIXED (memory-safe DataLoader)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # ✅ FIX
        pin_memory=(device.type == "cuda")  # ✅ FIX
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=0,  # ✅ FIX
        pin_memory=(device.type == "cuda")  # ✅ FIX
    )

    # =========================
    # MODEL
    # =========================
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for param in model.features[:-4].parameters():
        param.requires_grad = False

    for param in model.features[-4:].parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    model = model.to(device)

    # =========================
    # TRAINING SETUP
    # =========================
    weights = torch.tensor([1.0, 1.3]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    scaler = torch.amp.GradScaler("cuda")

    # =========================
    # TRAINING LOOP
    # =========================
    best_f1 = 0

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Training Loss: {train_loss:.4f}")

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)

        precision = precision_score(all_labels, all_preds, pos_label=1)
        recall = recall_score(all_labels, all_preds, pos_label=1)
        f1 = f1_score(all_labels, all_preds, pos_label=1)

        accuracy = (cm[0][0] + cm[1][1]) / cm.sum()

        print("Confusion Matrix:\n", cm)
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision (AI): {precision:.4f}")
        print(f"Recall (AI): {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "model/best_model.pth")
            print("🔥 Best model saved")

    with open("model/class_names.json", "w") as f:
        json.dump(train_data.classes, f)

    torch.save(model.state_dict(), "model/model.pth")
    print("\nFinal model saved ✅")


if __name__ == "__main__":
    main()