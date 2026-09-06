import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model


MODEL_PATH = "model/isl_gru_model.keras"
LABELS_PATH = "model/labels.txt"

SEQUENCE_LENGTH = 30
FEATURES = 63

GRU_CONFIDENCE = 0.70

STABLE_COUNT = 4



print("\n==============================================")
print("       LOADING GRU SEQUENCE MODEL")
print("==============================================")

try:

    gru_model = load_model(
        MODEL_PATH,
        compile=False
    )

    print("GRU model loaded successfully!")

except Exception as e:

    print("GRU model loading error:")
    print(e)
    raise SystemExit



try:

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        labels = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print("Labels loaded successfully:")
    
    for i, label in enumerate(labels):
        print(f"{i}: {label}")

except Exception as e:

    print("Label loading error:")
    print(e)
    raise SystemExit




if gru_model.output_shape[-1] != len(labels):

    print("\nERROR:")
    print(
        f"Model has {gru_model.output_shape[-1]} outputs "
        f"but labels.txt has {len(labels)} labels."
    )

    raise SystemExit


if gru_model.input_shape[1:] != (
    SEQUENCE_LENGTH,
    FEATURES
):

    print("\nERROR:")
    print(
        "Model input shape does not match detector."
    )

    print(
        "Expected:",
        (SEQUENCE_LENGTH, FEATURES)
    )

    print(
        "Found:",
        gru_model.input_shape[1:]
    )

    raise SystemExit


print("\nModel input shape:", gru_model.input_shape)
print("Model output shape:", gru_model.output_shape)



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.6,

    min_tracking_confidence=0.6
)



camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("\nERROR: Camera could not be opened!")

    hands.close()

    raise SystemExit



sequence = deque(
    maxlen=SEQUENCE_LENGTH
)




prediction_history = deque(
    maxlen=STABLE_COUNT
)




current_prediction = "Waiting..."

current_confidence = 0.0

communication = []

last_accepted = ""




def extract_landmarks(hand_landmarks):

    features = []

    for landmark in hand_landmarks.landmark:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        features,
        dtype=np.float32
    )




def build_sentence(words):

    if not words:

        return ""


   

    cleaned = []

    for word in words:

        if not cleaned or cleaned[-1] != word:

            cleaned.append(word)


   

    if cleaned == ["i", "want", "water"]:
        return "I want water."

    if cleaned == ["i", "want", "food"]:
        return "I want food."

    if cleaned == ["i", "want", "drink"]:
        return "I want a drink."

    if cleaned == ["hello"]:
        return "Hello!"

    if cleaned == ["thank_you"]:
        return "Thank you."

    if cleaned == ["sorry"]:
        return "Sorry."

    if cleaned == ["help"]:
        return "Help me."

    if cleaned == ["come"]:
        return "Come."

    if cleaned == ["happy"]:
        return "I am happy."

    if cleaned == ["no"]:
        return "No."


    text = " ".join(cleaned)

    return text.capitalize() + "."



def label_to_word(label):

    mapping = {

        "COME": "come",

        "FOOD": "food",

        "HAPPY": "happy",

        "HELP": "help",

        "I": "i",

        "NO": "no",

        "SORRY": "sorry",

        "THANK YOU": "thank_you",

        "WANT": "want",

        "WATER": "water"
    }

    return mapping.get(
        label,
        label.lower()
    )




def accept_gesture(label):

    global last_accepted

    word = label_to_word(label)

    if not word:

        return


    # Don't immediately accept the same gesture again

    if word == last_accepted:

        return


    communication.append(word)

    last_accepted = word

    print(
        f"Accepted gesture: {label}"
    )

    print(
        f"Communication: {build_sentence(communication)}"
    )



print("\n==============================================")
print("      ISL GRU SEQUENCE DETECTOR")
print("==============================================")

print("\nClasses:")

for i, label in enumerate(labels):

    print(
        f"{i}: {label}"
    )


print("\nControls:")
print("C = Clear communication")
print("Q = Quit")

print("\nShow one gesture continuously.")
print(
    f"The model will collect {SEQUENCE_LENGTH} frames."
)

print("==============================================\n")



while True:

    success, frame = camera.read()

    if not success:

        print("Camera frame error.")

        break



    frame = cv2.flip(
        frame,
        1
    )



    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )



    results = hands.process(
        rgb
    )



    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]



        features = extract_landmarks(
            hand
        )


      

        if len(features) == FEATURES:

            sequence.append(
                features
            )


 

        mp_drawing.draw_landmarks(

            frame,

            hand,

            mp_hands.HAND_CONNECTIONS
        )



        if len(sequence) == SEQUENCE_LENGTH:

            try:

                input_data = np.array(
                    sequence,
                    dtype=np.float32
                )


               

                input_data = np.expand_dims(
                    input_data,
                    axis=0
                )


               

                prediction = gru_model.predict(

                    input_data,

                    verbose=0
                )[0]



                class_index = int(
                    np.argmax(prediction)
                )


                current_confidence = float(
                    prediction[class_index]
                )


                # Get corresponding label

                if class_index < len(labels):

                    predicted_label = labels[
                        class_index
                    ]

                else:

                    predicted_label = "Unknown"



                if (
                    current_confidence
                    >= GRU_CONFIDENCE
                ):

                    current_prediction = predicted_label

                    prediction_history.append(
                        predicted_label
                    )


               

                    if (
                        len(prediction_history)
                        == STABLE_COUNT
                    ):

                        first_prediction = (
                            prediction_history[0]
                        )


                        if all(

                            p == first_prediction

                            for p in prediction_history

                        ):

                          

                            accept_gesture(
                                first_prediction
                            )


                          

                            sequence.clear()

                            prediction_history.clear()

                else:

                    current_prediction = "Unknown"

                    prediction_history.clear()


            except Exception as e:

                print(
                    "GRU prediction error:",
                    e
                )

                current_prediction = "Error"

                current_confidence = 0.0

                prediction_history.clear()



    else:

        current_prediction = "No Hand"

        current_confidence = 0.0

        sequence.clear()

        prediction_history.clear()

        

        last_accepted = ""


  

    height, width = frame.shape[:2]


  

    cv2.rectangle(

        frame,

        (10, 10),

        (width - 10, 55),

        (0, 0, 0),

        -1
    )


    cv2.putText(

        frame,

        "ISL GRU Communication System",

        (20, 42),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2
    )



    cv2.putText(

        frame,

        f"Gesture: {current_prediction}",

        (20, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 255, 0),

        2
    )



    cv2.putText(

        frame,

        f"Confidence: {current_confidence * 100:.1f}%",

        (20, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )



    cv2.putText(

        frame,

        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",

        (20, 150),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )



    cv2.putText(

        frame,

        f"Stable: {len(prediction_history)}/{STABLE_COUNT}",

        (20, 180),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )



    cv2.rectangle(

        frame,

        (15, 205),

        (width - 15, 330),

        (30, 30, 30),

        -1
    )


    cv2.putText(

        frame,

        "Communication:",

        (25, 235),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )


    sentence = build_sentence(
        communication
    )


    # Keep display readable

    if len(sentence) > 55:

        sentence = sentence[-55:]


    cv2.putText(

        frame,

        sentence,

        (25, 280),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

        (0, 255, 0),

        2
    )



    cv2.putText(

        frame,

        "C = Clear    Q = Quit",

        (20, 365),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )



    cv2.imshow(

        "ISL GRU Sequence Detector",

        frame
    )



    key = cv2.waitKey(1) & 0xFF


    # Clear

    if key == ord("c"):

        communication.clear()

        sequence.clear()

        prediction_history.clear()

        current_prediction = "Waiting..."

        current_confidence = 0.0

        last_accepted = ""

        print(
            "Communication cleared."
        )


   

    elif key == ord("q"):

        break



camera.release()

hands.close()

cv2.destroyAllWindows()



print("\n==============================================")
print("             PROGRAM STOPPED")
print("==============================================")

print(
    "Final sentence:"
)

print(
    build_sentence(communication)
)

print("==============================================")
