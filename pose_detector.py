import os
import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIGGER_PATH = os.path.join(BASE_DIR, "static", "trigger_result.txt")
current_score = 0

POSE_ANGLES = {
    "Mountain Pose": {
        "left_arm": (160, 200),
        "right_arm": (160, 200),
        "left_leg": (170, 190),
        "right_leg": (170, 190)
    },
    "Tree Pose": {
        "left_leg": (40, 100),   # lifted leg
        "right_leg": (160, 190), # standing leg
        "left_arm": (140, 200),
        "right_arm": (140, 200)
    },
    "Warrior Pose": {
    "left_arm": (165, 185),
    "right_arm": (165, 185),
    "right_leg": (80, 130),    # now this is bent
    "left_leg": (160, 190)     # now this is straight
}


}

def generate_frames(target_pose="Mountain Pose"):
    global current_score
    pose_hold_frames = 0
    required_frames = 15  # adjust to 30 for stricter matching

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("❌ Cannot access webcam. Please check camera permission.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

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
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        }

        for pose_name, expected_angles in POSE_ANGLES.items():
            match = True
            for joint, (min_angle, max_angle) in expected_angles.items():
                angle = angles.get(joint, None)
                if angle is None or not (min_angle <= angle <= max_angle):
                    match = False
                    break
            if match:
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

            # Display pose and target
            cv2.putText(image, f'Target: {target_pose}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f'Pose: {label}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show angles
            y_offset = 90
            for part, angle in angles.items():
                cv2.putText(image, f'{part}: {int(angle)}', (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Pose holding logic with visual countdown
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

def get_score():
    global current_score
    return current_score

