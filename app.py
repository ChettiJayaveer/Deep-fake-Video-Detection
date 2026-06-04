import os
import secrets
from PIL import Image
from flask import Flask, render_template, url_for, flash, redirect, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import torch

# Import model and preprocessing
from model_files.preprocessing import preprocess_video, allowed_file
from model_files.model_architecture import load_detector_model

# ---------------- Configuration ----------------
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess-this-secret'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///users.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    PROFILE_PICS_FOLDER = 'static/profile_pics'
    MODEL_PATH = 'model_files/best_model.pth'

# ---------------- App Initialization ----------------
app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'neon-yellow'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROFILE_PICS_FOLDER'], exist_ok=True)

# ---------------- Database Model ----------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    city = db.Column(db.String(100), nullable=True)
    password_hash = db.Column(db.String(200), nullable=False)
    profile_pic_filename = db.Column(db.String(120), nullable=False, default='default.png')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- Helper Functions ----------------
def save_profile_picture(picture_file):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(picture_file.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.config['PROFILE_PICS_FOLDER'], picture_fn)

    output_size = (125, 125)
    i = Image.open(picture_file)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

# ---------------- Model Loading ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    detector_model = load_detector_model(app.config['MODEL_PATH'], DEVICE)
    print(f"✅ Deepfake Detector Model Loaded Successfully on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    detector_model = None

# ---------------- Routes ----------------
@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html", title="Welcome")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("detect"))

    if request.method == "POST":
        full_name = request.form.get("full_name")
        username = request.form.get("username")
        email = request.form.get("email")
        city = request.form.get("city")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        profile_pic = request.files.get("profile_pic")

        if password != confirm_password:
            flash("⚠ Passwords do not match!", "neon-pink")
            return redirect(url_for("signup"))

        # Check if username or email exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            flash("⚠ Username or Email already exists!", "neon-pink")
            return redirect(url_for("signup"))

        if profile_pic and profile_pic.filename != "":
            picture_file = save_profile_picture(profile_pic)
        else:
            picture_file = "default.png"

        # Create and save new user
        new_user = User(
            full_name=full_name,
            username=username,
            email=email,
            city=city,
            profile_pic_filename=picture_file,
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("✅ Account created successfully! Please log in.", "neon-blue")
        return redirect(url_for("login"))

    return render_template("signup.html", title="Sign Up")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("detect"))

    if request.method == "POST":
        username_or_email = request.form.get("username")
        password = request.form.get("password")
        remember = request.form.get("remember") == "on"

        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"✅ Welcome back, {user.username}!", "neon-cyan")
            next_page = request.args.get("next")
            return redirect(next_page) if next_page else redirect(url_for("detect"))
        else:
            flash("❌ Invalid credentials. Try again.", "neon-pink")

    return render_template("login.html", title="Login")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("🔒 Logged out successfully.", "neon-yellow")
    return redirect(url_for("index"))

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", title="User Profile")

@app.route("/detect")
@login_required
def detect():
    return render_template("detect.html", title="Detector Terminal", result=None)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "video_file" not in request.files:
        return jsonify({"error": "No video file part.", "category": "neon-pink"}), 400

    file = request.files["video_file"]

    if file.filename == "":
        return jsonify({"error": "No video selected for uploading.", "category": "neon-yellow"}), 400

    if file and allowed_file(file.filename):
        if detector_model is None:
            return jsonify({"error": "Model not loaded. Cannot process request.", "category": "neon-pink"}), 503

        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)

        try:
            input_tensor = preprocess_video(video_path)
            input_tensor = input_tensor.to(DEVICE)

            with torch.no_grad():
                output = detector_model(input_tensor)

            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            real_prob = probabilities[0] * 100
            fake_prob = probabilities[1] * 100

            prediction = "FAKE" if fake_prob > real_prob else "REAL"
            confidence = round(max(fake_prob, real_prob), 2)

            # ✅ Fix: convert all numpy.float32 to Python float
            return jsonify({
                "success": True,
                "prediction": prediction,
                "confidence": float(confidence),
                "real_prob": float(round(real_prob, 2)),
                "fake_prob": float(round(fake_prob, 2))
            }), 200

        except Exception as e:
            return jsonify({"error": f"Server error: {e}", "category": "neon-pink"}), 500
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
    else:
        return jsonify({"error": "Invalid file type. Please upload MP4 or AVI.", "category": "neon-pink"}), 400

# ---------------- Run App ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create DB tables if they don't exist
    app.run(debug=True)
