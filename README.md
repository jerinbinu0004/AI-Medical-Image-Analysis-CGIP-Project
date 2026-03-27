# 🧠 AI Medical Image Analysis System

## 🔍 Overview

This project is a **multi-disease medical image analysis system** that uses deep learning models to detect various diseases from medical images. Users can upload images through a web interface, and the system predicts the disease along with a confidence score.

The system is designed to be **modular, scalable, and easy to extend** with new diseases.

---

## 🏥 Diseases Supported

* 🫁 Pneumonia (Chest X-ray)
* 🦠 Tuberculosis (TB)
* 🧫 Malaria (Cell Images)
* 🧠 Brain Tumor (MRI)
* 👁️ Glaucoma (Eye Images)
* 🦴 Bone Fracture (X-ray)

---

## ⚙️ Tech Stack

### Backend

* Python (Flask)
* TensorFlow / Keras
* OpenCV
* NumPy

### Frontend

* HTML
* CSS
* JavaScript

### Additional

* Node.js (for server-side support)
* JSON (for label mapping)

---

## 🧩 Project Structure

```
AI-Medical-Image-Analysis/
│
├── backend/
│   └── model/
│       ├── *_label.json
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│
├── app.py
├── server.js
├── requirements.txt
├── package.json
├── README.md
```

---

## 🚀 Features

* Upload medical images for prediction
* Multi-disease support in a single system
* Displays prediction with confidence score
* Clean and interactive UI
* Modular backend for easy extension

---

## 🧠 Model Details

* Binary classification models for:

  * Pneumonia, Tuberculosis, Malaria, Glaucoma, Bone Fracture
* Multi-class classification model for:

  * Brain Tumor (Glioma, Meningioma, Pituitary, No Tumor)
* Models trained using deep learning architectures (TensorFlow/Keras)

---

## 🖥️ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/ai-medical-image-analysis.git
cd ai-medical-image-analysis
```

---

### 2️⃣ Install Python dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Install Node dependencies

```
npm install
```

---

### 4️⃣ Run the backend

```
python app.py
```

---

### 5️⃣ Open in browser

```
http://localhost:5000
```

---

## 📸 Screenshots

*Add screenshots of your UI here*

---

## ⚠️ Note

Trained model files (`.h5`, `.keras`) are not included in this repository due to file size limitations. The repository contains the complete implementation and label mappings.

---

## 🎯 Future Enhancements

* Add more diseases
* Improve model accuracy
* Deploy as a cloud-based application
* Add real-time webcam analysis

---

## 👨‍💻 Contributors

* Your Name
* Your Teammate Name

---

## 📌 License

This project is for educational and research purposes.
