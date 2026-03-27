# app.py - Multi Disease Medical AI Backend with Glaucoma JSON Fix

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=''
)

CORS(app)

# ===============================
# Configuration
# ===============================

MODEL_DIR = 'model'

PNEUMONIA_MODEL_PATH = os.path.join(MODEL_DIR, 'pneumonia_model.h5')
PNEUMONIA_LABEL_PATH = os.path.join(MODEL_DIR, 'pneumonia_label.json')

FRACTURE_MODEL_PATH = os.path.join(MODEL_DIR, 'bone_fracture_model.h5')
FRACTURE_LABEL_PATH = os.path.join(MODEL_DIR, 'bone_label.json')

TB_MODEL_PATH = os.path.join(MODEL_DIR, 'tuberculosis_model.keras')
TB_LABEL_PATH = os.path.join(MODEL_DIR, 'tb_label.json')

MALARIA_MODEL_PATH = os.path.join(MODEL_DIR, 'malaria_model.h5')
MALARIA_LABEL_PATH = os.path.join(MODEL_DIR, 'malaria_label.json')

BRAIN_TUMOR_MODEL_PATH = os.path.join(MODEL_DIR, 'brain_tumor_model.keras')
BRAIN_TUMOR_LABEL_PATH = os.path.join(MODEL_DIR, 'brain_tumor_label.json')

GLAUCOMA_MODEL_PATH = os.path.join(MODEL_DIR, 'glaucoma_model.keras')
GLAUCOMA_LABEL_PATH = os.path.join(MODEL_DIR, 'glaucoma_label.json')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

models = {}
labels_map = {}

# ===============================
# Load Models
# ===============================

def load_models():
    global models, labels_map

    models['pneumonia'] = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)
    with open(PNEUMONIA_LABEL_PATH, 'r') as f:
        labels_map['pneumonia'] = json.load(f)
    print("Pneumonia model loaded")

    models['fracture'] = tf.keras.models.load_model(FRACTURE_MODEL_PATH)
    with open(FRACTURE_LABEL_PATH, 'r') as f:
        labels_map['fracture'] = json.load(f)
    print("Fracture model loaded")

    models['tuberculosis'] = tf.keras.models.load_model(TB_MODEL_PATH, compile=False)
    with open(TB_LABEL_PATH, 'r') as f:
        labels_map['tuberculosis'] = json.load(f)
    print("Tuberculosis model loaded")

    models['malaria'] = tf.keras.models.load_model(MALARIA_MODEL_PATH, compile=False)
    with open(MALARIA_LABEL_PATH, 'r') as f:
        labels_map['malaria'] = json.load(f)
    print("Malaria model loaded")

    models['brain_tumor'] = tf.keras.models.load_model(BRAIN_TUMOR_MODEL_PATH, compile=False)
    with open(BRAIN_TUMOR_LABEL_PATH, 'r') as f:
        labels_map['brain_tumor'] = json.load(f)
    print("Brain Tumor model loaded")

    models['glaucoma'] = tf.keras.models.load_model(GLAUCOMA_MODEL_PATH, compile=False)
    with open(GLAUCOMA_LABEL_PATH, 'r') as f:
        labels_map['glaucoma'] = json.load(f)
    print("Glaucoma model loaded")

    print("All models ready")

# ===============================
# Prediction Route
# ===============================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        disease_type = request.form.get('disease_type')

        print("Received disease_type:", disease_type)
        print("Available models:", models.keys())

        if disease_type not in models:
            return jsonify({'error': 'Invalid disease type'}), 400

        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        # ===============================
        # Resize Based on Disease
        # ===============================

        if disease_type == 'pneumonia':
            img = img.resize((224, 224))
        elif disease_type == 'fracture':
            img = img.resize((150, 150))
        elif disease_type == 'tuberculosis':
            img = img.resize((224, 224))
        elif disease_type == 'malaria':
            img = img.resize((128, 128))
        elif disease_type == 'brain_tumor':
            img = img.resize((224, 224))
        elif disease_type == 'glaucoma':
            img = img.resize((224, 224))

        img_array = np.array(img)

        # ===============================
        # Preprocessing Per Model
        # ===============================

        if disease_type == 'brain_tumor' or disease_type == 'glaucoma':
            img_array = preprocess_input(img_array)
        else:
            img_array = img_array.astype('float32') / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        model = models[disease_type]
        prediction = model.predict(img_array, verbose=0)

        print("Raw prediction array:", prediction)
        print("Prediction shape:", prediction.shape)

        # ===============================
        # Brain Tumor (Multi-Class)
        # ===============================

        if disease_type == 'brain_tumor':

            predicted_index = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100

            label_dict = labels_map['brain_tumor']
            index_to_label = {v: k for k, v in label_dict.items()}

            predicted_label = index_to_label[predicted_index].upper()

        # ===============================
        # Glaucoma (Binary - Sigmoid)
        # ===============================

        elif disease_type == 'glaucoma':

            prob = float(prediction[0][0])

            if prob >= 0.5:
                predicted_label = "GLAUCOMA"     # Referable Glaucoma
                confidence = prob * 100
            else:
                predicted_label = "NORMAL"    # Non Referable Glaucoma
                confidence = (1 - prob) * 100

        # ===============================
        # Other Binary Models
        # ===============================

        else:

            prob = float(prediction[0][0])

            if disease_type == 'pneumonia':
                if prob >= 0.5:
                    predicted_label = "PNEUMONIA"
                    confidence = prob * 100
                else:
                    predicted_label = "NORMAL"
                    confidence = (1 - prob) * 100

            elif disease_type == 'fracture':
                if prob >= 0.5:
                    predicted_label = "NOT_FRACTURED"
                    confidence = prob * 100
                else:
                    predicted_label = "FRACTURED"
                    confidence = (1 - prob) * 100

            elif disease_type == 'tuberculosis':
                if prob >= 0.5:
                    predicted_label = "TUBERCULOSIS"
                    confidence = prob * 100
                else:
                    predicted_label = "NORMAL"
                    confidence = (1 - prob) * 100

            elif disease_type == 'malaria':
                if prob <= 0.5:
                    predicted_label = "PARASITIZED"
                    confidence = (1 - prob) * 100
                else:
                    predicted_label = "UNINFECTED"
                    confidence = prob * 100

        return jsonify({
            'label': predicted_label,
            'confidence': f'{confidence:.2f}'
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500
    
# ===============================
# Serve Frontend
# ===============================

@app.route('/')
def index():
    return app.send_static_file('index.html')

# ===============================
# Run App
# ===============================

if __name__ == '__main__':
    try:
        load_models()
    except Exception as e:
        print("Model loading failed:", e)

    app.run(debug=False, host='0.0.0.0', port=3000)