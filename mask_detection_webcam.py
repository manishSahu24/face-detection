import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----- Paths -----
MODEL_PATH = "mask_detector_final.h5"

# Load mask classifier model
model = load_model(MODEL_PATH)

# Load OpenCV face detector (Haar cascade for simplicity)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_arr = preprocess_input(face_rgb.astype("float32"))
        face_arr = np.expand_dims(face_arr, axis=0)

        (mask_prob,) = model.predict(face_arr)[0]
        label = "Mask" if mask_prob >= 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label_text = f"{label}: {mask_prob:.2f}"

        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
