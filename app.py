import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

REGISTER_FOLDER = "images"
UPLOAD_FOLDER = "uploads"
ATTENDANCE_FILE = "Attendance.csv"

# Ensure required folders exist
for folder in [REGISTER_FOLDER, UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

known_encodings = []
known_names = []

# Load all known faces from the images folder
def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    for filename in os.listdir(REGISTER_FOLDER):
        img_path = os.path.join(REGISTER_FOLDER, filename)
        image = face_recognition.load_image_file(img_path)

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image for better detection
        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

        encoding = face_recognition.face_encodings(small_image, model="hog")

        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])  # Extract name from filename

load_known_faces()  # Load faces on startup

# Mark attendance for recognized faces
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)

    new_entry = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    print(f"✅ Attendance marked for {name} at {time_str}")

@app.route("/")
def home():
    return render_template("index.html")

# Register face using uploaded image
@app.route("/register/upload", methods=["POST"])
def register_upload():
    if "file" not in request.files or "name" not in request.form:
        return "No file or name received!", 400

    file = request.files["file"]
    name = request.form["name"].strip()

    if not name:
        return "Invalid name!", 400

    filename = f"{name.replace(' ', '_')}.jpg"
    file_path = os.path.join(REGISTER_FOLDER, filename)
    file.save(file_path)

    image = face_recognition.load_image_file(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    encodings = face_recognition.face_encodings(small_image, model="hog")

    print(f"🔍 Encodings for {name}: {encodings}")  # Debugging

    if not encodings:
        os.remove(file_path)
        return "❌ No face detected! Try again.", 400

    load_known_faces()
    return f"✅ Face registered successfully as {name}!"

# Register face using camera capture
@app.route("/register/camera", methods=["POST"])
def register_camera():
    if "file" not in request.files or "name" not in request.form:
        return "No file or name received!", 400

    file = request.files["file"]
    name = request.form["name"].strip()

    if not name:
        return "Invalid name!", 400

    filename = f"{name.replace(' ', '_')}.jpg"
    file_path = os.path.join(REGISTER_FOLDER, filename)
    file.save(file_path)

    image = face_recognition.load_image_file(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Step 1: Detect face locations first
    face_locations = face_recognition.face_locations(small_image, model="hog")
    encodings = face_recognition.face_encodings(small_image, face_locations)

    print(f"🔍 Face locations detected: {face_locations}")  # Debugging
    print(f"🔍 Encodings detected: {encodings}")  # Debugging

    if not encodings:
        os.remove(file_path)
        return "❌ No face detected! Try again.", 400

    load_known_faces()
    return f"✅ Face registered successfully as {name}!"

@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return "No file uploaded!", 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, "captured.jpg")
    file.save(file_path)

    load_known_faces()

    image = face_recognition.load_image_file(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Step 1: Detect face locations first
    face_locations = face_recognition.face_locations(small_image, model="hog")
    encodings = face_recognition.face_encodings(small_image, face_locations)

    print(f"🔍 Face locations detected: {face_locations}")  # Debugging
    print(f"🔍 Encodings detected: {encodings}")  # Debugging

    if not encodings:
        return "❌ No face detected!", 400

    encoding = encodings[0]
    matches = face_recognition.compare_faces(known_encodings, encoding)
    face_distances = face_recognition.face_distance(known_encodings, encoding)

    if any(matches):
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]
        mark_attendance(name)
        return f"✅ Face recognized: {name}. Attendance marked!"
    else:
        return "❌ Face not recognized!"

@app.route("/attendance")
def show_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        return df.to_html()
    return "No attendance records found."

if __name__ == "__main__":
    app.run(debug=True)
