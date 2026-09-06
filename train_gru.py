import os
import csv
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical


DATASET_PATH = "sequence_dataset"

SEQUENCE_LENGTH = 30
FEATURES = 63

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "isl_gru_model.keras"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "labels.txt"
)


X = []
y = []

print("======================================")
print("LOADING SEQUENCE DATASET")
print("======================================\n")

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: '{DATASET_PATH}' folder was not found.")
    exit()


for filename in sorted(os.listdir(DATASET_PATH)):

    if not filename.lower().endswith(".csv"):
        continue

    label = filename[:-4]

    file_path = os.path.join(
        DATASET_PATH,
        filename
    )

    file_sequence_count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.reader(file)

        for row in reader:

            try:
                values = [
                    float(value.strip())
                    for value in row
                ]

            except ValueError:
    
                continue


            if len(values) != SEQUENCE_LENGTH * FEATURES:
                continue

            sequence = np.array(
                values,
                dtype=np.float32
            )

            sequence = sequence.reshape(
                SEQUENCE_LENGTH,
                FEATURES
            )

            X.append(sequence)
            y.append(label)

            file_sequence_count += 1

    print(
        f"{label:12s} -> "
        f"{file_sequence_count} sequences"
    )


print("\n======================================")
print("DATASET SUMMARY")
print("======================================")

print(
    "Total sequences:",
    len(X)
)

classes_found = sorted(set(y))

print(
    "Number of classes:",
    len(classes_found)
)

print(
    "Classes:",
    classes_found
)


if len(X) == 0:

    print(
        "\nERROR: No valid sequences were found."
    )

    print(
        "Each CSV row must contain exactly "
        "1890 values (30 × 63)."
    )

    exit()



counts = Counter(y)

print("\nSequences per class:")

for label in sorted(counts):

    print(
        f"{label:12s}: {counts[label]}"
    )


X = np.array(
    X,
    dtype=np.float32
)


encoder = LabelEncoder()

y_integer = encoder.fit_transform(y)

print(
    "\nEncoded labels:",
    list(encoder.classes_)
)


X_train, X_test, y_train_int, y_test_int = train_test_split(

    X,
    y_integer,

    test_size=0.20,

    random_state=42,

    stratify=y_integer
)


y_train = to_categorical(
    y_train_int,
    num_classes=len(encoder.classes_)
)

y_test = to_categorical(
    y_test_int,
    num_classes=len(encoder.classes_)
)


print("\n======================================")
print("DATA SPLIT")
print("======================================")

print(
    "Training sequences:",
    len(X_train)
)

print(
    "Testing sequences:",
    len(X_test)
)

model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES
        )
    ),

    GRU(
        128,
        return_sequences=True
    ),

    Dropout(0.3),

    GRU(
        64
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        len(encoder.classes_),
        activation="softmax"
    )

])


model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


print("\n======================================")
print("MODEL")
print("======================================\n")

model.summary()


print("\n======================================")
print("GRU TRAINING STARTED")
print("======================================\n")


history = model.fit(

    X_train,

    y_train,

    epochs=50,

    batch_size=16,

    validation_data=(
        X_test,
        y_test
    ),

    verbose=1

)
loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)
print("\n======================================")
print("FINAL TEST RESULT")
print("======================================")

print(
    "TEST ACCURACY:",
    round(accuracy * 100, 2),
    "%"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model.save(
    MODEL_PATH
)


with open(
    LABEL_PATH,
    "w",
    encoding="utf-8"
) as file:

    for label in encoder.classes_:

        file.write(
            label + "\n"
        )


print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Labels:",
    LABEL_PATH
)


print("\nTesting model reload...")

try:

    test_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model reload successful!"
    )

    print(
        "Input shape:",
        test_model.input_shape
    )

    print(
        "Output shape:",
        test_model.output_shape
    )

except Exception as e:

    print(
        "\nERROR while reloading model:"
    )

    print(e)

    exit()

print("\n======================================")
print("GRU MODEL READY!")
print("======================================")

print("\nYour classes:")

for i, label in enumerate(encoder.classes_):

    print(
        f"{i}: {label}"
    )

print("\nNow run your sequence detector.")