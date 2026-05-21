import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)

import numpy as np
import cv2

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps

from predict import predict_disease
from ayurveda import AYURVEDA_SUGGESTIONS
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = "cutisai_secure_2026"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cutisai.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

@app.before_request
def load_user():
    user_id = session.get("user_id")
    g.user = db.session.get(User, user_id) if user_id else None


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not g.user:
            flash("Please login first")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

#Database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(200))
    theme = db.Column(db.String(10), default="Light")
    email_alerts = db.Column(db.Boolean, default=True)


class ClinicalData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)

    age = db.Column(db.String(10))
    gender = db.Column(db.String(20))
    area = db.Column(db.String(50))
    duration = db.Column(db.String(50))
    symptoms = db.Column(db.String(100))
    sensation = db.Column(db.String(100))

    image_path = db.Column(db.String(255))

    prediction = db.Column(db.String(100))
    confidence = db.Column(db.String(20))
    risk = db.Column(db.String(20))

    color_intensity = db.Column(db.Float)
    texture_irregularity = db.Column(db.Float)
    border_irregularity = db.Column(db.Float)
    asymmetry = db.Column(db.Float)
    pigmentation_variation = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

#Feature Extraction
def extract_features(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError("Image not readable")

    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "color": float(np.mean(img)),
        "texture": float(np.mean(cv2.Canny(gray, 100, 200))),
        "border": float(np.std(cv2.Canny(gray, 100, 200))),
        "asymmetry": float(abs(np.mean(gray[:, :112]) - np.mean(gray[:, 112:]))),
        "pigmentation": float(np.std(img))
    }

#Home Page
@app.route("/")
def home():
    return render_template("home.html")

#FAQ
faq = {
    "what is cutisai": "CutisAI is an AI-powered skin analysis platform that detects skin diseases using deep learning and provides early risk insights.",

    "how does cutisai work": "You upload a skin image → AI analyzes patterns → system predicts disease risk → gives insights and guidance.",

    "does cutisai diagnose skin cancer": "No. It only provides AI-based predictions. Always consult a doctor for diagnosis.",

    "how accurate is the ai": "The model is trained on dermatology datasets. Accuracy depends on image quality.",

    "is my data safe": "Yes. Your data is encrypted and protected. We do not share it without consent.",

    "how to use cutisai": "Register → Upload image → Get instant AI analysis → View report.",

    "is it free": "Basic analysis is available. Advanced features may require login."
}

#Disease Database
disease_info = {
    "melanoma": "⚠️ Critical: Dangerous skin cancer. Needs immediate medical attention.",
    "basal cell carcinoma": "⚠️ Action Required: Common skin cancer, slow-growing but serious.",
    "actinic keratosis": "⚠️ Pre-Cancerous: Can develop into cancer if untreated.",
    "melanocytic nevi": "🟢 Monitoring: Common moles, usually harmless.",
    "benign keratosis": "🟢 Benign: Non-cancerous skin growth.",
    "vascular lesions": "🟢 Benign: Blood vessel-related skin marks.",
    "dermatofibroma": "🟢 Benign: Small harmless skin nodules."
}

@app.route("/home-chat", methods=["POST"])
def home_chat():
    user_msg = request.json.get("message", "").lower().strip()

    if any(word in user_msg for word in ["hi", "hello", "hey"]):
        return jsonify({
            "reply": "Hello 👋 Welcome to CutisAI! Ask me about the platform, skin diseases, or how to use it."
        })

    if "what is this" in user_msg or "about" in user_msg:
        return jsonify({
            "reply": "CutisAI is an AI-based dermatology platform that helps detect skin diseases early using image analysis and provides risk insights."
        })

    for q, a in faq.items():
        if q in user_msg:
            return jsonify({"reply": a})

    for disease, info in disease_info.items():
        if disease in user_msg:
            return jsonify({"reply": info})

    if "help" in user_msg:
        return jsonify({
            "reply": "You can ask about:\n• What is CutisAI\n• Skin diseases\n• How to use the platform\n• Data safety"
        })

    return jsonify({
        "reply": "I can help with CutisAI info, FAQs, or skin diseases. Try asking something like 'What is melanoma?' or 'How does CutisAI work?'"
    })

#MAP MODEL - AYURVEDA
CLASS_TO_KEY = {
    "Melanocytic Nevus": "nv",
    "Melanoma": "mel",
    "Basal Cell Carcinoma": "bcc",
    "Actinic Keratosis": "akiec",
    "Benign Keratosis": "bkl",
    "Dermatofibroma": "df",
    "Vascular Lesion": "vasc"
}

#Deatils
@app.route("/detail")
@login_required
def detail():
    return render_template("detail.html", user=g.user)

#Analyze
@app.route("/analyze", methods=["POST"])
@login_required
def analyze():

    files = request.files.getlist("images")

    if not files or files[0].filename == "":
        flash("Please upload at least one image")
        return redirect(url_for("detail"))

    saved_images = []

    for file in files:
        filename = secure_filename(
            datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
        )
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        saved_images.append(filename)

    first_image_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_images[0])

    #model prediction
    result = predict_disease(first_image_path)

    top = result.get("predicted_disease", {})
    prediction = top.get("class", "Unknown").strip().title()
    confidence = float(top.get("probability", 0))

    features = extract_features(first_image_path) or {}

    color = float(features.get("color", 0))
    texture = float(features.get("texture", 0))
    border = float(features.get("border", 0))
    asymmetry = float(features.get("asymmetry", 0))
    pigmentation = float(features.get("pigmentation", 0))

    risk = (result.get("risk") or "Low").capitalize()

    record = ClinicalData(
        user_id=g.user.id,
        age=request.form.get("age"),
        gender=request.form.get("gender"),
        area=request.form.get("area"),
        duration=request.form.get("duration"),
        symptoms=request.form.get("symptoms"),
        sensation=request.form.get("sensation"),
        image_path=",".join(saved_images),
        prediction=prediction,
        confidence=f"{confidence:.2f}",
        risk=risk,
        color_intensity=color,
        texture_irregularity=texture,
        border_irregularity=border,
        asymmetry=asymmetry,
        pigmentation_variation=pigmentation
    )

    db.session.add(record)
    db.session.commit()

    return redirect(url_for("result", id=record.id))

user_context = {}

def detect_intent(msg):
    intents = {
        "greeting": ["hi", "hello", "hey"],
        "risk": ["risk", "danger", "serious"],
        "diet": ["diet", "food", "eat"],
        "herbs": ["herb", "ayurveda"],
        "treatment": ["care", "treatment", "cure"],
        "disease": ["what", "disease", "condition"],
        "symptoms": ["symptom", "sign"],
        "upload": ["upload", "photo"],
        "technical": ["how", "work", "ai"],
        "ayurveda_full": ["full ayurveda", "complete ayurveda"]
    }

    for intent, words in intents.items():
        if any(w in msg for w in words):
            return intent
    return "unknown"

def build_risk_response(ayur):
    risk = ayur.get("risk_level", "Unknown")

    if "very high" in risk.lower():
        level = "CRITICAL"
        advice = "Immediate medical consultation required."
    elif "high" in risk.lower():
        level = "HIGH"
        advice = "Consult a dermatologist soon."
    elif "moderate" in risk.lower():
        level = "MODERATE"
        advice = "Monitor regularly."
    else:
        level = "LOW"
        advice = "Generally safe, but observe changes."

    return f"{level}\nRisk Level: {risk}\n👉 {advice}"


def build_full_ayurveda(ayur):
    herbs = ", ".join(ayur.get("primary_herbs", [])) or "Not available"

    return f"""
AYURVEDIC ANALYSIS

Condition:
{ayur.get('disease', 'N/A')}

Dosha:
{ayur.get('targeted_dosha', 'N/A')}

Diet Plan:
{ayur.get('dietary_protocol', 'N/A')}

Herbs:
{herbs}

Care:
{ayur.get('care_tip', 'N/A')}

Research:
{ayur.get('research_basis', 'N/A')}
"""

def build_default(ayur, predicted_class):
    return f"""
Condition: {ayur.get('disease', predicted_class)}
Risk: {ayur.get('risk_level', 'Unknown')}

Quick Tip:
{ayur.get('care_tip', 'No care info available.')}

Ask:
• diet
• herbs
• treatment
• symptoms
"""

def get_followups(intent):
    followups = {
        "risk": "Ask about diet or treatment next.",
        "diet": "You can also ask about herbs.",
        "herbs": "Want full Ayurvedic plan?",
        "treatment": "Ask about risk level.",
        "default": "Try asking about diet, herbs, or symptoms."
    }
    return followups.get(intent, followups["default"])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "").lower().strip()
    predicted_class = data.get("prediction")
    confidence = data.get("confidence", 0.0)  # optional

    key = CLASS_TO_KEY.get(predicted_class)
    ayur = AYURVEDA_SUGGESTIONS.get(key, {})

    intent = detect_intent(user_msg)

    user_context["last_intent"] = intent
    user_context["last_condition"] = ayur.get("disease", predicted_class)

    if intent == "greeting":
        return jsonify({
            "reply": (
                "Hello! I am CutisAI Assistant.\n\n"
                "I can analyze your skin condition and guide you.\n\n"
                "Ask about:\n• Risk\n• Diet\n• Herbs\n• Treatment"
            )
        })

    elif intent == "risk":
        reply = build_risk_response(ayur)

    elif intent == "diet":
        reply = f"Diet Plan:\n{ayur.get('dietary_protocol', 'No data available.')}"

    elif intent == "herbs":
        herbs = ", ".join(ayur.get("primary_herbs", [])) or "No data"
        reply = f"Herbs:\n{herbs}"
    
    elif intent == "treatment":
        reply = f"Care:\n{ayur.get('care_tip', 'No care info available.')}"
        if ayur.get("warning"):
            reply += f"\n {ayur['warning']}"

    elif intent == "ayurveda_full":
        reply = build_full_ayurveda(ayur)

    elif intent == "symptoms":
        reply = (
            "Symptoms:\n"
            "• Asymmetry\n• Border irregularity\n• Color variation\n"
            "• Itching\n• Bleeding"
        )

    elif intent == "upload":
        reply = (
            "Upload Tips:\n"
            "• Good lighting\n• No blur\n• Close-up lesion\n• No filters"
        )

    elif intent == "technical":
        reply = (
            "CutisAI uses CNN (Deep Learning)\n"
            "trained on dermatology image datasets."
        )
        
    else:
        reply = build_default(ayur, predicted_class)

    if confidence and confidence < 0.6:
        reply += "\n\nLow confidence in prediction. Consider re-uploading image."

    reply += "\n\n" + get_followups(intent)

    return jsonify({"reply": reply})

#Result
@app.route("/result/<int:id>")
@login_required
def result(id):

    record = ClinicalData.query.get_or_404(id)

    images = record.image_path.split(",") if record.image_path else []

    confidence = float(record.confidence or 0)
    prediction_clean = (record.prediction or "Unknown").strip()
    risk = (record.risk or "Low").capitalize()

    first_image_path = (
        os.path.join(app.config["UPLOAD_FOLDER"], images[0])
        if images else None
    )

    all_predictions = []

    if first_image_path and os.path.exists(first_image_path):
        try:
            result_data = predict_disease(first_image_path)
            all_predictions = result_data.get("all_predictions", [])
        except Exception as e:
            print("Prediction Error:", e)

    PREDICTION_TO_KEY = {
        "Actinic Keratosis": "akiec",
        "Basal Cell Carcinoma": "bcc",
        "Benign Keratosis": "bkl",
        "Dermatofibroma": "df",
        "Melanoma": "mel",
        "Melanocytic Nevus": "nv",
        "Vascular Lesion": "vasc"
    }

    ayurveda_key = PREDICTION_TO_KEY.get(prediction_clean, None)
    ayurveda = AYURVEDA_SUGGESTIONS.get(ayurveda_key, {})

    image_params = {
        "asymmetry": round(float(record.asymmetry or 0), 3),
        "border_irregularity": round(float(record.border_irregularity or 0), 3),
        "color_variance": round(float(record.color_intensity or 0), 3),
        "texture_entropy": round(float(record.texture_irregularity or 0), 3),
        "pigmentation_variation": round(float(record.pigmentation_variation or 0), 3),
    }

    user = getattr(g, "user", None)

    return render_template(
        "result.html",
        record=record,
        images=images,
        prediction=prediction_clean,
        confidence=confidence,
        risk=risk,
        all_predictions=all_predictions,
        ayurveda=ayurveda,
        user=user,
        image_params=image_params,
        age=record.age,
        gender=record.gender,
        area=record.area,
        symptoms=record.symptoms,
        sensation=record.sensation,
        duration=record.duration
    )

#Dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    records = ClinicalData.query.filter_by(user_id=g.user.id).all()
    return render_template("dashboard.html", user=g.user, records=records)

@app.route('/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    record = ClinicalData.query.get(record_id)

    if record and record.user_id == g.user.id:
        db.session.delete(record)
        db.session.commit()

    return redirect(url_for('dashboard'))

#Authentication
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = User(
            fullname=request.form["fullname"],
            email=request.form["email"],
            password=generate_password_hash(request.form["password"])
        )
        db.session.add(user)
        db.session.commit()
        session["user_id"] = user.id
        return redirect(url_for("detail"))
    return render_template("register.html")

#Profile
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "profile":
            g.user.fullname = request.form.get("fullname")
            g.user.email = request.form.get("email")
            db.session.commit()
            flash("Profile updated successfully!")

        elif form_type == "password":
            new_pass = request.form.get("password")
            if new_pass:
                g.user.password = generate_password_hash(new_pass)
                db.session.commit()
                flash("Password changed successfully!")

        elif form_type == "preferences":
            g.user.email_alerts = 'email_alerts' in request.form
            db.session.commit()
            flash("Preferences saved!")

        elif form_type == "appearance":
            g.user.theme = request.form.get("theme")
            db.session.commit()
            flash(f"Theme set to {g.user.theme}")

        return redirect(url_for("profile"))

    return render_template("profile.html", user=g.user)

@app.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    # Delete clinical records first
    ClinicalData.query.filter_by(user_id=g.user.id).delete()
    # Delete user
    db.session.delete(g.user)
    db.session.commit()
    session.clear()
    flash("Account deleted successfully.")
    return redirect(url_for("home"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(email=request.form["email"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            session["user_id"] = user.id
            return redirect(url_for("dashboard"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    print("Server Started Successfully!")
    print("http://127.0.0.1:5000")
    app.run(debug=True)