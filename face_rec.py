import os
import cv2
import face_recognition
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
known_dir = os.path.join(BASE_DIR, 'known_faces')

known_encodings = []
known_names = []

# Load known faces (1 image per person)
for filename in os.listdir(known_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(known_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].lower()
            known_names.append(name)

last_matched_user = None

def generate_face_frames():
    global last_matched_user
    cap = cv2.VideoCapture(0)
    matched = False
    match_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not matched:
            for face_encoding in face_encodings:
                tolerance = 0.38  # super strict
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                print("Distances:", face_distances)  # Debug
                print("Best:", known_names[best_match_index], "| Distance:", face_distances[best_match_index])  # Debug

                if matches[best_match_index]:
                    last_matched_user = known_names[best_match_index]
                    matched = True
                    match_time = time.time()
                    print(f"✅ Matched: {last_matched_user} (distance: {face_distances[best_match_index]:.4f})")
                    break
                else:
                    print("❌ No confident match found")

        if matched and time.time() - match_time > 3:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def get_last_recognized_user():
    global last_matched_user
    user = last_matched_user
    last_matched_user = None
    return user
