import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
import torch.nn as nn

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORM (same as val)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# LOAD VAL DATA
# =========================
val_data = datasets.ImageFolder("data/val", transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

# =========================
# LOAD MODEL
# =========================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("model/model.pth", map_location=device))
model = model.to(device)
model.eval()

# =========================
# COLLECT LOGITS
# =========================
logits_list = []
labels_list = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)

        logits_list.append(outputs)
        labels_list.append(labels.to(device))

logits = torch.cat(logits_list)
labels = torch.cat(labels_list)

# =========================
# FIND BEST TEMPERATURE
# =========================
temperature = torch.nn.Parameter(torch.ones(1).to(device))

optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

def loss_fn():
    optimizer.zero_grad()
    scaled_logits = logits / temperature
    loss = F.cross_entropy(scaled_logits, labels)
    loss.backward()
    return loss

optimizer.step(loss_fn)

# Clamp temperature
temperature.data = torch.clamp(temperature.data, min=0.05, max=10)

T = temperature.item()
print(f"\n🔥 Best Temperature: {T:.4f}")

# SAVE 🔥
torch.save(temperature.detach(), "model/temperature.pth")
print("✅ Temperature saved")