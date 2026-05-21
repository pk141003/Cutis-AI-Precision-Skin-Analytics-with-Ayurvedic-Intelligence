import numpy as np
import tensorflow as tf
from PIL import Image
from ayurveda import AYURVEDA_SUGGESTIONS

model = tf.keras.models.load_model("model/model.h5", compile=False)

dummy = np.zeros((1, 224, 224, 3))
model.predict(dummy, verbose=0)

IMG_SIZE = 224

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus",
    "Vascular Lesion"
]

def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_disease(path):
    img = preprocess(path)
    preds = model.predict(img)[0]

    results = []
    for i, p in enumerate(preds):
        results.append({
            "class": CLASS_NAMES[i],
            "probability": round(float(p * 100), 2)
        })

    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    return {
        "predicted_disease": results[0],
        "all_predictions": results
    }