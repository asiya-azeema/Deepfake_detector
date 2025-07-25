# 🛡️ AI Deep Forge – Deepfake Detection System

AI Deep Forge is a deep learning-based project that detects **deepfake images and videos** by analyzing facial frames using a pretrained **Xception CNN** model. The system processes videos, extracts frames, trains a binary classifier, and deploys the model using a **Flask web app**.

---

## 📌 Project Highlights

- 🔍 Detects real vs. fake faces from video frames
- 🧠 Uses transfer learning with Xception (Keras)
- 🎞️ Frame extraction and preprocessing from FaceForensics++
- 🌐 Simple Flask web interface for prediction
- ✅ Real-time prediction with confidence scores

---

## 🧠 Data Science & AI Skills Demonstrated

- **Data Collection & Preprocessing:** Video frame extraction, normalization
- **Model Building:** Xception-based binary image classification
- **Evaluation:** Accuracy, real-time prediction analysis
- **Deployment:** Flask-based web interface
- **Use Case:** AI forensics, ethical AI, misinformation mitigation

---

## 🗂️ Project Structure

```
📁 ai-deep-forge/
├── ai deep forge3.py          # Core logic: frame extraction, training, prediction
├── app.py                     # Flask web application
├── test_flask.py              # Flask test route
├── static/uploads/            # Folder to store uploaded images
├── templates/index.html       # Web UI template (if present)
├── deepfake_model.h5          # Trained model (generated after training)
```

---

## ⚙️ How It Works

1. **Extract Frames**  
   From original and manipulated videos using OpenCV.

2. **Train Deep Learning Model**  
   Using Xception with a binary output (Real or Fake).

3. **Deploy Model**  
   A Flask app takes user-uploaded images and predicts if they are fake.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install tensorflow keras opencv-python flask numpy tqdm
```

### 2. Run Frame Extraction & Model Training

```bash
python "ai deep forge3.py"
```

This will:
- Extract video frames
- Train the model
- Save the model as `deepfake_model.h5`

### 3. Run the Flask Web App

```bash
python app.py
```

Visit: `http://127.0.0.1:5000` to upload an image and check if it's fake or real.

---

## 🧪 Example Prediction

```
[INFO] Prediction: Fake (Confidence: 0.94)
```

---

## 📚 Dataset Used

- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)  
  (Original + Deepfakes + FaceSwap + Face2Face)

---

## 👤 Author

**Asiyamath Azeema**  
Dept. of Artificial Intelligence & Data Science  
Bearys Institute of Technology  
USN: 4BP22AD008

---

## 📘 License

This project is intended for educational and research purposes only. Always use deepfake detection responsibly.
