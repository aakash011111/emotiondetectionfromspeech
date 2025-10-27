from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import librosa
import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Simulated user credentials
USER_CREDENTIALS = {
    "user": "password"
}

# Load model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

UPLOAD_FOLDER = "static/uploads"
PLOT_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

def extract_mfcc_segment(y, sr, start, duration=1.0):
    """Extract MFCC from a segment of audio."""
    segment = y[int(start * sr):int((start + duration) * sr)]
    mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40).T, axis=0)
    return np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)

def predict_over_time(audio_path, step=0.5, duration=1.0):
    """Predict emotions over time using sliding windows."""
    y, sr = librosa.load(audio_path)
    total_duration = librosa.get_duration(y=y, sr=sr)
    times = []
    emotions = []
    all_predictions = []

    for start in np.arange(0, total_duration - duration, step):
        features = extract_mfcc_segment(y, sr, start, duration)
        prediction = model.predict(features, verbose=0)[0]
        predicted_emotion = EMOTIONS[np.argmax(prediction)]
        times.append(start)
        emotions.append(predicted_emotion)
        all_predictions.append(prediction)

    # Calculate average probabilities
    avg_predictions = np.mean(all_predictions, axis=0)
    
    return times, emotions, avg_predictions

def plot_emotion_timeline(times, emotions, out_path):
    """Plot emotion timeline."""
    emotion_to_int = {e: i for i, e in enumerate(EMOTIONS)}
    numeric_emotions = [emotion_to_int[e] for e in emotions]

    plt.figure(figsize=(12, 3))
    plt.scatter(times, numeric_emotions, c='blue', alpha=0.6)
    plt.yticks(list(emotion_to_int.values()), list(emotion_to_int.keys()))
    plt.xlabel("Time (s)")
    plt.ylabel("Emotion")
    plt.title("Emotion Timeline")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_emotion_distribution(emotions, out_path):
    """Plot bar chart for emotion distribution."""
    counts = {e: emotions.count(e) for e in EMOTIONS}
    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.title("Emotion Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/authenticate", methods=["POST"])
def authenticate():
    username = request.form.get("username")
    password = request.form.get("password")

    if USER_CREDENTIALS.get(username) == password:
        session["username"] = username
        return redirect(url_for("index"))
    else:
        error = "Invalid credentials. Please try again."
        return render_template("login.html", error=error)

@app.route("/index")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("username"))

@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return jsonify({"error": "Unauthorized access"}), 401

    audio_file = request.files.get("audio_file")
    if not audio_file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded file
    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)

    try:
        times, emotions, avg_predictions = predict_over_time(file_path)

        timeline_path = os.path.join(PLOT_FOLDER, f"timeline_{filename}.png")
        dist_path = os.path.join(PLOT_FOLDER, f"dist_{filename}.png")

        plot_emotion_timeline(times, emotions, timeline_path)
        plot_emotion_distribution(emotions, dist_path)

        # Most frequent emotion
        final_emotion = max(set(emotions), key=emotions.count)

        # Create emotion probabilities dictionary
        emotion_probabilities = {
            emotion: float(prob) for emotion, prob in zip(EMOTIONS, avg_predictions)
        }

        return jsonify({
            "predicted_emotion": final_emotion,
            "emotion_probabilities": emotion_probabilities,
            "timeline_img": f"/{timeline_path}",
            "dist_img": f"/{dist_path}",
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)