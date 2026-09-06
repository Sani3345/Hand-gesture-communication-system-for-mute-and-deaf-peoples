import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model


model = load_model("model/isl_gru_model.keras")


with open("model/labels.txt", "r") as file:
    labels = [line.strip() for line in file.readlines()]

print("Labels:", labels)



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    exit()


sequence = deque(maxlen=30)

prediction = "Waiting..."

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

   

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for landmark in hand.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        sequence.append(landmarks)

        # Draw hand
        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    else:
        # No hand detected
        sequence.append([0] * 63)

    

    if len(sequence) == 30:

        input_data = np.array(sequence, dtype=np.float32)

        input_data = np.expand_dims(input_data, axis=0)

        prediction_result = model.predict(
            input_data,
            verbose=0
        )

        predicted_index = np.argmax(prediction_result)

        confidence = prediction_result[0][predicted_index]

        prediction = labels[predicted_index]

        
        if confidence < 0.60:
            prediction = "Unknown"

        cv2.putText(
            frame,
            f"Sign: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Collecting frames...",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "ISL Real-Time Sign Detection",
        frame
    )

    # Q = quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
hands.close()