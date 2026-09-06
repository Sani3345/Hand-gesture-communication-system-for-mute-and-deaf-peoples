import cv2
import mediapipe as mp
import csv
import os
import time

SEQUENCE_LENGTH = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs("sequence_dataset", exist_ok=True)

label = input("Enter sign name: ").strip().upper()

file_name = f"sequence_dataset/{label}.csv"

file = open(file_name, "a", newline="")
writer = csv.writer(file)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    exit()

print()
print(f"Sign: {label}")
print("Press SPACE to record one complete movement.")
print("Press Q to quit.")
print()

sample_count = 0

while True:

    success, frame = camera.read()

    if not success:
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

    cv2.putText(
        frame,
        f"{label}  Samples: {sample_count}/50",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Record | Q = Quit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("ISL Movement Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:

        print("Recording...")

        sequence = []

        for i in range(SEQUENCE_LENGTH):

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

                mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

            else:
                sequence.append([0] * 63)

            cv2.putText(
                frame,
                "RECORDING...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

            cv2.imshow("ISL Movement Dataset", frame)

            cv2.waitKey(1)

            time.sleep(0.05)

        if len(sequence) == SEQUENCE_LENGTH:

            # One complete movement = one row
            flattened = []

            for frame_landmarks in sequence:
                flattened.extend(frame_landmarks)

            writer.writerow(flattened)
            file.flush()

            sample_count += 1

            print(f"Saved movement sample {sample_count}/50")

    elif key == ord("q"):
        break

camera.release()
file.close()
cv2.destroyAllWindows()
hands.close()

print()
print(f"Finished {label}: {sample_count} samples")