# 🧠 AI Image Detector

An end-to-end deep learning system that detects whether an image is **AI-generated or real**, built using PyTorch, FastAPI, and Gradio.

---

## 🚀 Features

* 🔍 Detect AI-generated vs Real images
* 📊 Confidence-based prediction with uncertainty handling
* 🔥 Grad-CAM visualization (model explainability)
* ⚡ Fast API backend using FastAPI
* 🎨 Interactive UI using Gradio
* 📂 Batch image analysis support

---

## 🧠 Model Details

* Architecture: **EfficientNet-B0**
* Framework: **PyTorch**
* Transfer Learning (partial layer freezing)
* Techniques used:

  * Data Augmentation (flip, rotation, blur, grayscale)
  * Custom Gaussian Noise Injection
  * Label Smoothing
  * Class Imbalance Handling
  * Mixed Precision Training (AMP)
  * Cosine Learning Rate Scheduler
  * Gradient Clipping

---

## 📊 Dataset

The dataset consists of both AI-generated and real-world images collected from multiple sources.

### 🤖 AI Images

* DiffusionDB dataset (Hugging Face)
* Additional samples from Kaggle datasets

### 🌍 Real Images

* COCO dataset
* Public images collected from:

  * Google Images
  * Pexels
  * Instagram (public content)
  * Kaggle datasets

### 📈 Dataset Split

| Category    | Training | Validation |
| ----------- | -------- | ---------- |
| AI Images   | 3072     | 746        |
| Real Images | 2966     | 763        |

* Total images used: **~7,500+**
* Dataset is balanced across classes

📊 Model Performance

Best model (Epoch 7):

Accuracy: 92.84%
Precision: 0.9338
Recall: 0.9240
F1 Score: 0.9289

### 🛠️ Data Processing

* Images were cleaned and filtered
* Converted to consistent format
* Organized into structured folders (`train/`, `val/`)
* Custom scripts used for dataset preparation

> Note: A subset of a much larger dataset was used due to computational constraints.


---

## 🧩 Project Structure

```
ai-image-detector/
│
├── app.py                  # Gradio UI
├── train.py                # Model training
├── predict.py             # Inference logic
├── requirements.txt
│
├── api/
│   └── main.py            # FastAPI backend
│
├── data/
│   ├── ai/                # AI dataset scripts
│   └── real/              # Real dataset scripts
│
├── utils/                 # Helper functions
├── model/                 # Saved models
```

---

## ⚙️ How It Works

1. Train model using `train.py`
2. Run FastAPI backend:

   ```
   uvicorn api.main:app --reload
   ```
3. Launch UI:

   ```
   python app.py
   ```
4. Upload image → Get prediction + heatmap

---

## 📡 API Endpoints

### `/predict`

* Input: Image file
* Output:

  * Label
  * Confidence
  * AI / Real probability
  * Optional heatmap

### `/explain`

* Returns Grad-CAM visualization

---

## 🧠 Prediction Logic

* Uses softmax probabilities
* Calculates:

  * Confidence
  * Margin between classes

### Decision Rules:

* Low confidence → **Uncertain**
* High confidence → AI / Real classification

---

## 🔥 Explainability (Grad-CAM)

The model highlights important regions in the image:

* 🔴 High attention
* 🟡 Medium attention
* 🔵 Low attention

---

## ⚠️ Disclaimer

Predictions are not always 100% accurate.
Model performance depends on dataset quality and generalization.

---

## 👨‍💻 Author

**Lankesh**

---

## 🚀 Future Improvements

* 📈 Train on larger and more diverse datasets to improve generalization and robustness
* 🧠 Fine-tune model architecture and hyperparameters for better accuracy
* 🔥 Enhance Grad-CAM visualization for more precise and interpretable heatmaps
* 🎨 Improve user interface with better UX and real-time feedback
* ⚡ Optimize inference speed for faster predictions
* 🐳 Containerize the application using Docker for easy deployment
* ☁️ Deploy the system on cloud platforms (AWS / GCP / Azure)
* 📱 Extend support for web/mobile applications


---

