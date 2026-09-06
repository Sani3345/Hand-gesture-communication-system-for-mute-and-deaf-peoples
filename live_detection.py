import cv2
import mediapipe as mp
import pickle


with open("model/sign_model.pkl", "rb") as file:
    model = pickle.load(file)

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
    print("Camera could not be opened")
    exit()

print("Live detection started!")
print("Show your sign. Press Q to quit.")

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "No Hand"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Extract 21 landmarks
        landmarks = []

        for landmark in hand.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

       
        prediction = model.predict([landmarks])[0]

       
        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


    cv2.putText(
        frame,
        prediction,
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("ISL Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
hands.close()