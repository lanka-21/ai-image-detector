#import os
#import sys

# Add project root to Python path
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append(BASE_DIR)
import gradio as gr
from PIL import Image
import time

from inference.predict import predict_image

import requests
import io
import base64
import numpy as np
import cv2
PREDICT_URL = "http://127.0.0.1:8000/predict?explain=true"
EXPLAIN_URL = "http://127.0.0.1:8000/explain"
# =========================
# SAFE WRAPPER
# =========================
def safe_predict(img):
    try:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {
            "file": ("image.jpg", img_bytes, "image/jpeg")
        }

        # 🔥 SINGLE API CALL
        response = requests.post(PREDICT_URL, files=files)

        if response.status_code != 200:
            raise Exception("API error")

        data = response.json()
        pred_data = data["prediction"]

        heatmap_b64 = pred_data.get("heatmap")

        heatmap_img = None

        if heatmap_b64:
            img_bytes = base64.b64decode(heatmap_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            heatmap_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if heatmap_img is not None:
                heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

        pred_data["heatmap"] = heatmap_img

        print("Heatmap shape:", heatmap_img.shape if heatmap_img is not None else None)

        return pred_data

    except Exception as e:
        print("ERROR:", e)
        return {
            "label": "Error",
            "confidence": 0,
            "ai_prob": 0,
            "real_prob": 0,
            "margin": 0,
            "heatmap": None
        }

def format_output(result):
    label = result["label"]
    confidence = result["confidence"]
    ai = result["ai_prob"]
    real = result["real_prob"]
    margin = result["margin"]

    # Uncertainty logic (move BEFORE return)
    if confidence < 0.65 or margin < 0.15:
        label = "Uncertain"

    color = {
        "AI Generated": "#ff4d4d",
        "Real Image": "#4CAF50",
        "Uncertain": "#f1c40f"
    }.get(label, "#333")

    confidence_level = (
        "High" if confidence > 0.85 else
        "Medium" if confidence > 0.65 else
        "Low"
    )

    return f"""
    <div style="
        padding:20px;
        border-radius:12px;
        background: var(--background-fill-secondary);
        color: var(--body-text-color);
        border:1px solid var(--border-color-primary);
        box-shadow:0 2px 10px rgba(0,0,0,0.1);
    ">
        <h2 style="color:{color};">{label}</h2>

        <p><b>Confidence:</b> {confidence*100:.2f}% ({confidence_level})</p>

        <div style="margin-top:10px;">
            <p>AI Probability</p>
            <div style="background: var(--background-fill-primary); border-radius:10px;">
                <div style="width:{ai*100}%;background:#ff4d4d;height:10px;border-radius:10px;"></div>
            </div>
        </div>

        <div style="margin-top:10px;">
            <p>Real Probability</p>
            <div style="background: var(--background-fill-primary); border-radius:10px;">
                <div style="width:{real*100}%;background:#4CAF50;height:10px;border-radius:10px;"></div>
            </div>
        </div>
    </div>
    """

    # Improved uncertainty logic (NOW WORKS)
    if confidence < 0.65 or margin < 0.15:
        label = "Uncertain"

    # Color logic
    color = {
        "AI Generated": "#ff4d4dc0",
        "Real Image": "#4CAF50",
        "Uncertain": "#f1c40f"
    }.get(label, "white")

    # Confidence level
    confidence_level = (
        "High" if confidence > 0.85 else
        "Medium" if confidence > 0.65 else
        "Low"
    )

    return f"""
    <div style="
        padding:25px;
        border-radius:15px;
        background:linear-gradient(145deg,#1e1e1e,#2c2c2c);
        box-shadow:0 0 15px rgba(0,0,0,0.5);
        color:white;
    ">
        <h2 style="color:{color};">{label}</h2>

        <p><b>Confidence:</b> {confidence*100:.2f}% ({confidence_level})</p>

        <div style="margin-top:10px;">
            <p>AI Probability</p>
            <div style="background:#333;border-radius:10px;">
                <div style="width:{ai*100}%;background:#ff4d4d;padding:5px;border-radius:10px;"></div>
            </div>
        </div>

        <div style="margin-top:10px;">
            <p>Real Probability</p>
            <div style="background:#333;border-radius:10px;">
                <div style="width:{real*100}%;background:#4CAF50;padding:5px;border-radius:10px;"></div>
            </div>
        </div>
    </div>
    """
# =========================
# SINGLE IMAGE
# =========================
def predict_single(image):
    start = time.time()

    result = safe_predict(image)

    end = time.time()

    output = format_output(result)   # ✅ changed
    latency = f"{(end - start):.2f} sec"

    explanation = explain_heatmap(result)

    return output, latency, result.get("heatmap", None), explanation


# =========================
# MULTIPLE IMAGES
# =========================
def predict_multiple(files):
    results = []

    start = time.time()

    for f in files:
        img = Image.open(f).convert("RGB")
        result = safe_predict(img)

        results.append(
            f"{f.name} → {result['label']} ({result['confidence']*100:.1f}%)"
        )

    end = time.time()

    return "\n".join(results), f"{(end - start):.2f} sec"


# =========================
# heatmap
# =========================
def explain_heatmap(result):
    label = result["label"]

    if "AI" in label:
        return "⚠️ The model focused on unnatural patterns (textures, shapes, or inconsistencies)."

    elif "Real" in label:
        return "✅ The model focused on natural structures like faces, objects, and consistent textures."

    else:
        return "🤔 The model is uncertain. The highlighted regions are not clearly informative."
    



# =========================
# UI
# =========================
with gr.Blocks(title="AI Detector") as app:

    # ✅ NEW PRO HEADER
    gr.HTML("""
<div style="
    text-align:center;
    padding:60px 20px;
    background: radial-gradient(circle at top, #1e293b, #020617);
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 25px 80px rgba(0,0,0,0.8);
">

    <div style="
        display:inline-block;
        padding:6px 14px;
        border-radius:999px;
        background:rgba(59,130,246,0.15);
        color:#60a5fa;
        font-size:13px;
        margin-bottom:20px;
        border:1px solid rgba(59,130,246,0.3);
    ">
        AI Powered Detection System
    </div>

    <h1 style="
        font-size:56px;
        font-weight:900;
        margin:0;
        color:white;
    ">
        AI Detector World
    </h1>

    <div style="
        margin-top:15px;
        display:flex;
        justify-content:center;
        gap:20px;
        flex-wrap:wrap;
    ">
        <span style="color:#cbd5f5;">
            Built by <b style="color:white;">Lankesh</b>
        </span>

        <span style="
            padding:8px 16px;
            border-radius:10px;
            background:rgba(255,204,0,0.1);
            border:1px solid rgba(255,204,0,0.4);
            color:#facc15;
        ">
            ⚠️ Predictions may not always be accurate
        </span>
    </div>

</div>
""")
    # =========================
# PROFESSIONAL INFO SECTION
# =========================
    gr.HTML("""
    <div style="
        margin-top:15px;
        padding:16px;
        border-radius:14px;
        background:linear-gradient(145deg,#020617,#0f172a);
        border:1px solid rgba(255,255,255,0.08);
    ">

    <h3 style="color:#e2e8f0;margin-bottom:10px;">
    🧠 How the AI looked at your image
    </h3>

    <div style="
        display:flex;
        gap:12px;
        flex-wrap:wrap;
    ">

        <!-- RED -->
        <div style="
            flex:1;
            min-width:140px;
            padding:12px;
            border-radius:12px;
            background:rgba(255,0,0,0.08);
            border:1px solid rgba(255,0,0,0.3);
            transition:0.3s;
        ">
            <div style="font-size:18px;">🔴</div>
            <b style="color:#f87171;">Important Area</b><br>
            <span style="font-size:13px;color:#cbd5f5;">
                AI is looking here very carefully 👀
            </span>
        </div>

        <!-- YELLOW -->
        <div style="
            flex:1;
            min-width:140px;
            padding:12px;
            border-radius:12px;
            background:rgba(255,204,0,0.08);
            border:1px solid rgba(255,204,0,0.3);
        ">
            <div style="font-size:18px;">🟡</div>
            <b style="color:#facc15;">Somewhat Important</b><br>
            <span style="font-size:13px;color:#cbd5f5;">
                AI is checking this, but not fully
            </span>
        </div>

        <!-- BLUE -->
        <div style="
            flex:1;
            min-width:140px;
            padding:12px;
            border-radius:12px;
            background:rgba(59,130,246,0.08);
            border:1px solid rgba(59,130,246,0.3);
        ">
            <div style="font-size:18px;">🔵</div>
            <b style="color:#60a5fa;">Not Important</b><br>
            <span style="font-size:13px;color:#cbd5f5;">
                AI is mostly ignoring this
            </span>
        </div>

    </div>

    <div style="
        margin-top:12px;
        padding:10px;
        border-radius:10px;
        background:rgba(255,255,255,0.03);
        color:#94a3b8;
        font-size:13px;
    ">
    💡 Think of it like this:  
    AI is “looking” at the image just like you do — focusing more on important parts.
    </div>

    </div>
    """)

    gr.Markdown("---")

# =========================
# TABS WITH CLEAN DESIGN
# =========================
    with gr.Tabs():

        # =========================
        # SINGLE IMAGE
        # =========================
        with gr.Tab("🔍 Analyze Image"):
            with gr.Row():

                # LEFT SIDE
                with gr.Column():
                    image_input = gr.Image(
                        type="pil",
                        label="📤 Upload Image",
                        height=300
                    )

                # RIGHT SIDE
                with gr.Column():
                    output_text = gr.HTML(label="🧠 AI Decision")
                    latency = gr.Textbox(label="⚡ Inference Time")
                    heatmap_output = gr.Image(label="🔥 AI Attention Heatmap")
                    heatmap_explanation = gr.Markdown(label="📊 Explanation")

                    gr.HTML("""
                    <div style="
                        margin-top:15px;
                        padding:16px;
                        border-radius:14px;
                        background:linear-gradient(145deg,#020617,#0f172a);
                        border:1px solid rgba(255,255,255,0.08);
                    ">

                    <h3 style="color:#e2e8f0;margin-bottom:10px;">
                    🧠 How the AI looked at your image
                    </h3>

                    <div style="
                        margin-bottom:12px;
                        padding:10px;
                        border-radius:10px;
                        background:rgba(255,255,255,0.03);
                        color:#94a3b8;
                        font-size:13px;
                    ">
                    ⚠️ These colors are <b>NOT in your image</b>.<br>
                    They are added by the AI to show <b>where it is focusing</b>.
                    </div>

                    <div style="display:flex;gap:12px;flex-wrap:wrap;">

                        <div style="
                            flex:1;
                            min-width:140px;
                            padding:12px;
                            border-radius:12px;
                            background:rgba(255,0,0,0.08);
                            border:1px solid rgba(255,0,0,0.3);
                        ">
                            🔴 <b>AI focused here</b><br>
                            <span style="font-size:13px;color:#cbd5f5;">
                            This part influenced the decision most
                            </span>
                        </div>

                        <div style="
                            flex:1;
                            min-width:140px;
                            padding:12px;
                            border-radius:12px;
                            background:rgba(255,204,0,0.08);
                            border:1px solid rgba(255,204,0,0.3);
                        ">
                            🟡 <b>AI checked here</b><br>
                            <span style="font-size:13px;color:#cbd5f5;">
                            This part helped a little
                            </span>
                        </div>

                        <div style="
                            flex:1;
                            min-width:140px;
                            padding:12px;
                            border-radius:12px;
                            background:rgba(59,130,246,0.08);
                            border:1px solid rgba(59,130,246,0.3);
                        ">
                            🔵 <b>AI ignored this</b><br>
                            <span style="font-size:13px;color:#cbd5f5;">
                            This part was not important
                            </span>
                        </div>

                    </div>

                    <div style="
                        margin-top:12px;
                        padding:10px;
                        border-radius:10px;
                        background:rgba(34,197,94,0.08);
                        border:1px solid rgba(34,197,94,0.3);
                        color:#86efac;
                        font-size:13px;
                    ">
                    💡 Think of it like this:  
                    The AI is showing a <b>highlight map</b> of where it “looked” — just like how you focus on important parts when deciding.
                    </div>

                    </div>
                    """)

            btn1 = gr.Button("🚀 Analyze Image", variant="primary")

            btn1.click(
                fn=predict_single,
                inputs=image_input,
                outputs=[output_text, latency, heatmap_output, heatmap_explanation]
            )

        # =========================
        # MULTIPLE IMAGE
        # =========================
        with gr.Tab("📂 Batch Analysis"):
            file_input = gr.File(file_types=["image"], file_count="multiple")
            multi_output = gr.Textbox(label="📊 Results")
            multi_latency = gr.Textbox(label="⚡ Processing Time")

            btn2 = gr.Button("📊 Run Batch Analysis")

            btn2.click(
                fn=predict_multiple,
                inputs=file_input,
                outputs=[multi_output, multi_latency]
            )
    # =========================
app.launch()