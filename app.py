from flask import Flask, render_template, Response, redirect, url_for, jsonify, session, request
from pose_detector import generate_frames, get_score, TRIGGER_PATH
import random
import os
from face_rec import get_last_recognized_user, generate_face_frames

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'
current_score = 0

# List of available poses
POSES = [
    {"name": "Mountain Pose", "img": "poses/mountain.png"},
    {"name": "Tree Pose", "img": "poses/Tree_pose.png"},
    {"name": "Warrior Pose", "img": "poses/Warrior_pose.png"}
]

# Landing page
@app.route('/')
def home():
    return render_template('index.html')

# Face scanner entry page
@app.route('/face-scan')
def face_scan():
    session.pop('user', None)  # 🧼 clear any existing user before scanning
    return render_template('face_scan.html')

# Check user match from session or global
@app.route('/check_user')
def check_user():
    user = session.get('user')
    if not user:
        user = get_last_recognized_user()
        if user:
            session['user'] = user
    if user:
        return jsonify({"status": "matched", "user": user})
    return jsonify({"status": "waiting"})

# Greeting route after match
@app.route('/greeting')
def greeting():
    user = session.get('user')
    if not user:
        return redirect(url_for('face_scan'))
    return render_template('face_res.html', user=user)

# Main dashboard after face matched
@app.route('/choice')
def choice():
    return render_template('choice.html')

# Breathing route
@app.route('/breathing')
def breathing():
    affirmations = [
        "✨ You are calm, grounded, and enough.",
        "🌿 In this moment, you are exactly where you need to be.",
        "🧘 Breathe in strength, breathe out tension.",
        "💫 You are capable, you are peaceful, you are strong."
    ]
    return render_template('breathing.html', random_affirmation=random.choice(affirmations), duration=15)

# Live pose detector game
@app.route('/live-detector')
def live_detector():
    with open(TRIGGER_PATH, "w") as f:
        f.write("")  # clear trigger file
    chosen_pose = random.choice(POSES)
    session['target_pose'] = chosen_pose["name"]
    session['pose_img'] = chosen_pose["img"]
    return render_template('detector_live.html', pose_name=chosen_pose["name"], pose_img=chosen_pose["img"])

# Handle webcam stream for pose detection
@app.route('/video_feed')
def video_feed():
    pose_name = session.get('target_pose', 'Mountain Pose')
    return Response(generate_frames(pose_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# Webcam feed for face recognition
@app.route('/video_feed_face')
def video_feed_face():
    return Response(generate_face_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Trigger checker for pose game
@app.route('/check_trigger')
def check_trigger():
    try:
        with open(TRIGGER_PATH, "r") as f:
            if "done" in f.read():
                return jsonify({"status": "done"})
    except FileNotFoundError:
        pass
    return jsonify({"status": "waiting"})

# Result screen
@app.route('/result')
def result():
    global current_score
    current_score = get_score()
    with open(TRIGGER_PATH, "w") as f:
        f.write("")
    return render_template('detector_game_result.html', score=current_score)

# Mini games
@app.route('/tic-tac-toe')
def tic_tac_toe():
    return render_template('tic_tac_toe.html')

@app.route('/joke-game')
def joke_game():
    return render_template('joke_game.html')

if __name__ == '__main__':
    app.run(debug=True)