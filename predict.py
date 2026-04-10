import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import json

from inference.gradcam import GradCAM, overlay_heatmap
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pth")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
# =========================
# DEVICE (GPU SUPPORT 🔥)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TRANSFORM (MATCH TRAINING ✅)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# LOAD CLASS NAMES
# =========================
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)   # ['real', 'ai'] usually

# =========================
# LOAD MODEL (SAFE LOAD ✅)
# =========================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model = model.to(device)
model.eval()

target_layer = model.features[-2]
gradcam = GradCAM(model, target_layer)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(input_data, return_heatmap=False):

    if isinstance(input_data, str):
        img = Image.open(input_data).convert("RGB")
    else:
        img = input_data.convert("RGB")

    img = transform(img).unsqueeze(0).to(device)

    # ---- PREDICTION ----
    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()

    # ---- PROBABILITIES ----
    real_prob = probs[0][class_names.index("real")].item()
    ai_prob   = probs[0][class_names.index("ai")].item()

    confidence = max(ai_prob, real_prob)
    margin = abs(ai_prob - real_prob)

    # ---- DECISION ----
    if confidence < 0.65 or margin < 0.15:
        label = "Uncertain"
    elif confidence > 0.90:
        label = "AI Generated (High Confidence)" if ai_prob > real_prob else "Real Image (High Confidence)"
    else:
        label = "AI Generated (Medium Confidence)" if ai_prob > real_prob else "Real Image (Medium Confidence)"

    # ---- BASE RESULT ----
    result = {
        "label": label,
        "confidence": float(confidence),
        "ai_prob": float(ai_prob),
        "real_prob": float(real_prob),
        "margin": float(margin)
    }

    # =========================
    # 🔥 HEATMAP PART (ONLY IF NEEDED)
    # =========================
    if return_heatmap:
        try:
            cam = gradcam.generate(img, predicted_class)

            if cam is None:
                print("❌ GradCAM returned None")
                result["heatmap"] = None
                return result

            print("✅ GradCAM OK:", cam.shape)

            # prepare image
            img_disp = img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            img_disp = (img_disp * [0.229,0.224,0.225]) + [0.485,0.456,0.406]
            img_disp = (img_disp * 255).clip(0,255).astype("uint8")

            heatmap_img = overlay_heatmap(img_disp, cam)

            # encode to base64
            import cv2, base64, numpy as np

            heatmap = np.array(heatmap_img)

            if heatmap.dtype != np.uint8:
                heatmap = (heatmap * 255).clip(0,255).astype("uint8")

            success, buffer = cv2.imencode(".jpg", heatmap)

            if success:
                heatmap_base64 = base64.b64encode(buffer).decode("utf-8")
                result["heatmap"] = heatmap_base64
            else:
                result["heatmap"] = None

        except Exception as e:
            print("🔥 GradCAM ERROR:", str(e))
            result["heatmap"] = None

    return result
'''
    # Uncertain zone
    if confidence < 0.55 or margin < 0.10:
        return f"Uncertain ({confidence*100:.2f}%)"

    # AI detection (slight bias but controlled)
    if ai_prob > real_prob:
        return f"AI Generated ({ai_prob*100:.2f}%)"
    else:
        return f"Real Image ({real_prob*100:.2f}%)" '''

if __name__ == "__main__":
    test_folder = r"L:\PYTHONNNN\projects\camera\test_images"

    for filename in os.listdir(test_folder):
        path = os.path.join(test_folder, filename)

        # skip non-images
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        result = predict_image(path)

        print(f"{filename} → {result['label']} ({result['confidence']*100:.2f}%)")
