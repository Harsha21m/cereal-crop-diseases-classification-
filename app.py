import base64
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pymysql
# from src.predict import predict_from_bytes # Removing this to use local logic
from src.treatments import treatments
from src.localization import localization
from datetime import datetime
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model

pymysql.install_as_MySQLdb()

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

# ---------- DATABASE CONFIG ----------
# Load password from environment variable
password = quote_plus(os.getenv("DB_PASSWORD", ""))
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://root:{password}@localhost/cereal_crop_db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ---------- LOAD MODELS ----------
print("Loading models...")
# Load uploaded-image model (your old CNN)
try:
    old_model = load_model("models/crop_model.h5")
    print("✔ Loaded crop_model.h5")
except Exception as e:
    print(f"❌ Error loading crop_model.h5: {e}")

# Load real-time model (MobileNetV2)
try:
    mobilenet_model = load_model("models/mobilenet_model.h5")
    print("✔ Loaded mobilenet_model.h5")
except Exception as e:
    print(f"❌ Error loading mobilenet_model.h5: {e}")

# Load Label Encoder
try:
    label_encoder = joblib.load("models/label_encoder.joblib")
    print("✔ Loaded label_encoder.joblib")
except Exception as e:
    print(f"❌ Error loading label_encoder.joblib: {e}")


# ---------- HELPER FUNCTIONS ----------

def preprocess_uploaded_image(img_bytes):
    """Preprocess for the OLD CNN model (crop_model)"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_camera_image(img_bytes):
    """Preprocess for the MobileNet model (Center Crop + Resize)"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MobileNet was likely trained on RGB if using standard preprocessing, but let's check. Assuming standard.

    # Crop center region (reduces background noise)
    h, w, _ = img.shape
    # Take center 60% (0.2 to 0.8)
    crop = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    # Resize for MobileNet
    crop = cv2.resize(crop, (128, 128))
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=0)

    return crop

def predict_with_old_model(img_bytes):
    img = preprocess_uploaded_image(img_bytes)
    preds = old_model.predict(img)[0]
    idx = np.argmax(preds)
    label = label_encoder.inverse_transform([idx])[0]
    confidence = float(preds[idx])
    return label, confidence

def predict_with_mobilenet(img_bytes):
    img = preprocess_camera_image(img_bytes)
    preds = mobilenet_model.predict(img)[0]
    idx = np.argmax(preds)
    label = label_encoder.inverse_transform([idx])[0]
    confidence = float(np.max(preds))
    return label, confidence


# ---------- USER MODEL ----------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------- ROUTES ----------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('predict_page'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'warning')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


@app.route('/predict-page')
@login_required
def predict_page():
    return render_template('predict.html')


@app.route('/camera-page')
@login_required
def camera_page():
    return render_template('camera.html')


@app.route('/history')
@login_required
def history():
    if current_user.username == "admin":  # simple admin check
        records = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).all()
    else:
        records = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('history.html', records=records)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Route for Uploaded Images -> Uses OLD MODEL (crop_model.h5)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        
        # Use OLD MODEL
        label, confidence = predict_with_old_model(img_bytes)

        # Save prediction history
        new_record = PredictionHistory(
            user_id=current_user.id,
            image_name=file.filename,
            prediction=label,
            confidence=float(confidence)
        )
        db.session.add(new_record)
        db.session.commit()

        selected_language = request.form.get('language', 'en') # Get language from form, default to English
        treatment_key = treatments.get(label, "Unknown")
        treatment = localization.get(selected_language, {}).get(treatment_key, "No treatment information available.")
        
        # Convert image bytes to base64 so it can be displayed in browser
        encoded_image = base64.b64encode(img_bytes).decode('utf-8')

        return jsonify({
            'label': label,
            'confidence': round(confidence * 100, 2),
            'treatment': treatment,
            'image_data': encoded_image
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-camera', methods=['POST'])
@login_required
def predict_camera_route():
    """Route for Live Camera -> Uses MobileNet (mobilenet_model.h5)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        # Use MobileNet
        label, confidence = predict_with_mobilenet(img_bytes)

        # Confidence Threshold Check
        if confidence < 0.9:
            label = "Not a leaf"
            treatment = "No crop detected or confidence too low."
        else:
            treatment_key = treatments.get(label, "Unknown")
            # Default to English for camera for now, or could add language selector to camera page
            treatment = localization.get('en', {}).get(treatment_key, "No treatment information available.")

        # Save prediction history (Optional, but good for tracking)
        new_record = PredictionHistory(
            user_id=current_user.id,
            image_name="camera_capture.jpg",
            prediction=label,
            confidence=float(confidence)
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            'label': label,
            'confidence': round(confidence * 100, 2),
            'treatment': treatment
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_name = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PredictionHistory {self.id}>"


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Host='0.0.0.0' allows access from other devices on the network
        app.run(host='0.0.0.0', port=5000, debug=True)
