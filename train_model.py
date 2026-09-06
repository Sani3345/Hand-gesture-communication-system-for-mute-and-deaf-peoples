import csv
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file_name = "dataset/sign_data.csv"

X = []
y = []

with open(file_name, "r") as file:
    reader = csv.reader(file)

    next(reader)  

    for row in reader:
        features = [float(value) for value in row[:-1]]
        label = row[-1]

        X.append(features)
        y.append(label)

print("Total samples:", len(X))
print("Total classes:", len(set(y)))
print("Classes:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Model save karo
os.makedirs("model", exist_ok=True)

with open("model/sign_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as:")
print("model/sign_model.pkl")