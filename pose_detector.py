import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIGGER_PATH = os.path.join(BASE_DIR, "static", "trigger_result.txt")
current_score = 0

# Define poses and their expected joint angles
POSE_ANGLES = {
    "Mountain Pose": {
        "left_arm": (150, 210),   # was too strict before
        "right_arm": (150, 210),
        "left_leg": (160, 200),
        "right_leg": (160, 200)
    },
    "Tree Pose": {
        "left_leg": (30, 110),   # wider range for balance
        "right_leg": (150, 200),
        "left_arm": (130, 210),
        "right_arm": (130, 210)
    },
    "Warrior Pose": {
        "left_arm": (150, 195),
        "right_arm": (150, 195),
        "right_leg": (70, 140),
        "left_leg": (150, 195)
    }
}


# Reusable angle calculator
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# 🎥 Webcam-based local detection
def generate_frames(target_pose="Mountain Pose"):
    global current_score
    pose_hold_frames = 0
    required_frames = 15

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("❌ Cannot access webcam. Please check camera permissions.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    def classify_pose(landmarks):
        angles = {
            "left_arm": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
            "right_arm": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
            "left_leg": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
            "right_leg": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        }

        for pose_name, expected_angles in POSE_ANGLES.items():
            if all(min_a <= angles.get(joint, 0) <= max_a
                   for joint, (min_a, max_a) in expected_angles.items()):
                return pose_name, angles
        return "Unknown", angles

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            label = "No Pose"
            angles = {}
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                label, angles = classify_pose(landmarks)

            cv2.putText(image, f'Target: {target_pose}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f'Pose: {label}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            y_offset = 90
            for part, angle in angles.items():
                cv2.putText(image, f'{part}: {int(angle)}', (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            if label == target_pose:
                pose_hold_frames += 1
            else:
                pose_hold_frames = max(pose_hold_frames - 1, 0)

            print(f"[DEBUG] Held: {pose_hold_frames}/{required_frames} — Pose: {label}")

            if pose_hold_frames >= required_frames:
                current_score = 1
                print("✅ Pose held long enough — writing 'done' to trigger_result.txt")
                with open(TRIGGER_PATH, "w") as f:
                    f.write("done")
                break

            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# 🧠 Image-based pose classification (for Railway / browser uploads)
def detect_pose_from_image(base64_image, target_pose="Mountain Pose"):
    img_data = base64_image.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    mp_pose = mp.solutions.pose
    expected = POSE_ANGLES.get(target_pose, {})

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return {"matched": False, "pose": "Unknown"}

        landmarks = results.pose_landmarks.landmark
        angles = {
            "left_arm": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
            "right_arm": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
            "left_leg": calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
            "right_leg": calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        }

        for joint, (min_a, max_a) in expected.items():
            angle = angles.get(joint, 0)
            if not (min_a <= angle <= max_a):
                return {"matched": False, "pose": target_pose}

        return {"matched": True, "pose": target_pose}

# For result retrieval
def get_score():
    global current_score
    return current_score
