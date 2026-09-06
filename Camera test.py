import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera opened! Show your hand. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
    results = hands.process(rgb_frame)

    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Sign Language Hand Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
hands.close()