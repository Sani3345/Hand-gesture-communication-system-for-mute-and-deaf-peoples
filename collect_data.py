import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs("dataset", exist_ok=True)

file_name = "dataset/sign_data.csv"

file_exists = os.path.exists(file_name)

file = open(file_name, "a", newline="")
writer = csv.writer(file)

if not file_exists:
    header = []

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    header.append("label")
    writer.writerow(header)

label = input("Enter sign name: ")

print()
print(f"Collecting data for: {label}")
print("Show your sign to the camera.")
print("Press SPACE to save a sample.")
print("Press Q to quit.")
print()

camera = cv2.VideoCapture(0)

count = 0

while True:

    success, frame = camera.read()

    if not success:
        print("Camera error!")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        for landmark in hand.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        cv2.putText(
            frame,
            f"Samples: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        landmarks = []

        cv2.putText(
            frame,
            "Show your hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        f"Sign: {label}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:

        if landmarks:

            writer.writerow(landmarks + [label])
            file.flush()

            count += 1

            print(f"Saved sample {count}")

        else:
            print("No hand detected!")

    elif key == ord("q"):
        break

camera.release()
file.close()
cv2.destroyAllWindows()
hands.close()

print()
print(f"Done! {count} samples collected for '{label}'.")