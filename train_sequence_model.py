import os
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "sequence_dataset"

SEQUENCE_LENGTH = 30
FEATURES = 63

X = []
y = []

for filename in os.listdir(DATASET_PATH):

    if not filename.endswith(".csv"):
        continue

    label = filename[:-4]

    file_path = os.path.join(DATASET_PATH, filename)

    with open(file_path, "r") as file:

        reader = csv.reader(file)

        for row in reader:

            values = [float(x) for x in row]

            if len(values) == SEQUENCE_LENGTH * FEATURES:

                sequence = np.array(values)

                sequence = sequence.reshape(
                    SEQUENCE_LENGTH,
                    FEATURES
                )

                X.append(sequence)
                y.append(label)

print("Total sequences:", len(X))
print("Classes:", sorted(set(y)))

X = np.array(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_encoded = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = Sequential([

    GRU(
        128,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, FEATURES)
    ),

    Dropout(0.3),

    GRU(64),

    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\n==============================")
print("TEST ACCURACY:", round(accuracy * 100, 2), "%")
print("==============================")

os.makedirs("model", exist_ok=True)

model.save("model/isl_gru_model.keras")

with open("model/labels.txt", "w") as file:

    for label in encoder.classes_:
        file.write(label + "\n")

print("\nModel saved:")
print("model/isl_gru_model.keras")
print("model/labels.txt")