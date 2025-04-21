from flask import Flask, render_template, Response, redirect, url_for, jsonify, session, request
from pose_detector import generate_frames, get_score, TRIGGER_PATH, detect_pose_from_image
from face_rec import get_last_recognized_user, generate_face_frames
import face_recognition
import random
import os
import base64
import re
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'
current_score = 0

# Load known faces
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
known_dir = os.path.join(BASE_DIR, 'known_faces')
known_encodings = []
known_names = []

for filename in os.listdir(known_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        path = os.path.join(known_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0].lower())

# Pose options
POSES = [
    {"name": "Mountain Pose", "img": "poses/mountain.png"},
    {"name": "Tree Pose", "img": "poses/Tree_pose.png"},
    {"name": "Warrior Pose", "img": "poses/Warrior_pose.png"}
]

@app.route('/')
def home():
    global current_score
    current_score = 0
    return render_template('index.html')

@app.route('/face-scan')
def face_scan():
    session.pop('user', None)
    return render_template('face_scan.html')

@app.route('/api/face-recognition', methods=['POST'])
def api_face_recognition():
    data = request.get_json()
    img_data = re.sub('^data:image/.+;base64,', '', data.get('image', ''))
    try:
        image = Image.open(BytesIO(base64.b64decode(img_data)))
        rgb = np.array(image.convert('RGB'))

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                idx = matches.index(True)
                session['user'] = known_names[idx]
                return jsonify(success=True, user=known_names[idx])
    except Exception as e:
        print(f"[ERROR] Face recognition failed: {e}")
    return jsonify(success=False)

@app.route('/check_user')
def check_user():
    user = session.get('user') or get_last_recognized_user()
    if user:
        session['user'] = user
        return jsonify({"status": "matched", "user": user})
    return jsonify({"status": "waiting"})

@app.route('/greeting')
def greeting():
    user = session.get('user')
    if not user:
        return redirect(url_for('face_scan'))
    return render_template('face_res.html', user=user)

@app.route('/choice')
def choice():
    global current_score
    current_score = 0
    return render_template('choice.html')

@app.route('/breathing')
def breathing():
    affirmations = [
        "✨ You are calm, grounded, and enough.",
        "🌿 In this moment, you are exactly where you need to be.",
        "🧘 Breathe in strength, breathe out tension.",
        "💫 You are capable, you are peaceful, you are strong."
    ]
    return render_template('breathing.html', random_affirmation=random.choice(affirmations), duration=15)

@app.route('/live-detector')
def live_detector():
    with open(TRIGGER_PATH, "w") as f:
        f.write("")
    chosen_pose = random.choice(POSES)
    session['target_pose'] = chosen_pose["name"]
    session['pose_img'] = chosen_pose["img"]
    return render_template('detector_live.html', pose_name=chosen_pose["name"], pose_img=chosen_pose["img"])

@app.route('/pose_predict', methods=['POST'])
def pose_predict():
    data = request.get_json()
    base64_image = data.get('image')
    if not base64_image:
        return jsonify({"matched": False, "pose": "Unknown", "error": "No image received"})

    pose_name = session.get('target_pose', 'Mountain Pose')
    result = detect_pose_from_image(base64_image, pose_name)

    print(f"[POSE RESULT] Target: {pose_name}, Matched: {result['matched']}")
    if result["matched"]:
        with open(TRIGGER_PATH, "w") as f:
            f.write("done")

    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    pose_name = session.get('target_pose', 'Mountain Pose')
    return Response(generate_frames(pose_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_face')
def video_feed_face():
    return Response(generate_face_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_trigger')
def check_trigger():
    try:
        with open(TRIGGER_PATH, "r") as f:
            if "done" in f.read():
                return jsonify({"status": "done"})
    except FileNotFoundError:
        pass
    return jsonify({"status": "waiting"})

@app.route('/result')
def result():
    global current_score
    current_score = get_score()
    with open(TRIGGER_PATH, "w") as f:
        f.write("")
    return render_template('detector_game_result.html', score=current_score)

@app.route('/tic-tac-toe')
def tic_tac_toe():
    return render_template('tic_tac_toe.html')

@app.route('/joke-game')
def joke_game():
    return render_template('joke_game.html')

if __name__ == '__main__':
    app.run(debug=True)