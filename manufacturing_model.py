
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)
import joblib


# ==========================================
# STEP 1: LOAD MANUFACTURING DATASET
# ==========================================

print("Loading manufacturing dataset...")

dataset = fetch_ucirepo(id=601)

# Select only the five machine measurements
feature_columns = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]

X = dataset.data.features[feature_columns].copy()
y = dataset.data.targets["Machine failure"].astype(int)

print("\nDataset loaded:", X.shape)
print("Sensor variables:", X.columns.tolist())

print("\nMachine failure distribution:")
print(y.value_counts())


# ==========================================
# STEP 2: SPLIT THE DATASET
# ==========================================

# Reserve 20% for final testing
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Divide remaining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# ==========================================
# STEP 3: CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# STEP 4: TRAIN THE MODEL
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# STEP 5: VALIDATE DIFFERENT THRESHOLDS
# ==========================================

failure_index = list(model.classes_).index(1)

validation_probabilities = model.predict_proba(
    X_val
)[:, failure_index]

print("\n===== THRESHOLD COMPARISON =====")

for threshold in [0.50, 0.30]:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    print(f"\nThreshold: {threshold * 100:.0f}%")
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ))


# ==========================================
# STEP 6: SELECT DECISION THRESHOLD
# ==========================================

# Experimental threshold prioritising failure recall
threshold = 0.30

print("\nSelected threshold:", threshold)


# ==========================================
# STEP 7: TEST ON HELD-OUT DATA
# ==========================================

test_probabilities = model.predict_proba(
    X_test
)[:, failure_index]

test_predictions = (
    test_probabilities >= threshold
).astype(int)

print("\n===== FINAL MODEL EVALUATION =====")

accuracy = accuracy_score(
    y_test, test_predictions
)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

print("\nConfusion matrix:")

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

print("Correctly identified normal:", tn)
print("False alarms:", fp)
print("Missed failures:", fn)
print("Correctly detected failures:", tp)

if tp + fn > 0:
    detection_rate = tp / (tp + fn) * 100
else:
    detection_rate = 0

print(
    "\nFailure detection rate:",
    round(detection_rate, 2),
    "%"
)


# ==========================================
# STEP 8: SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "threshold": threshold
    },
    "manufacturing_model.joblib"
)

print("\nModel saved successfully!")
print("Manufacturing AI training completed!")
