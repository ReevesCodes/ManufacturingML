
import pandas as pd
import joblib

# STEP 1: Load our trained machine learning model
saved = joblib.load("manufacturing_model.joblib")

model = saved["model"]
features = saved["features"]

print("Manufacturing AI model loaded successfully!")

# STEP 2: Enter machine sensor readings
machine = pd.DataFrame(
    [[298.1, 308.6, 1551, 42.8, 25]],
    columns=features
)

print("\n===== MACHINE SENSOR READINGS =====")
print(machine)

# STEP 3: Get the predicted failure probability
failure_index = list(model.classes_).index(1)

probability = model.predict_proba(machine)[0][failure_index]

# STEP 4: Set our decision threshold
threshold = 0.30

# Predict failure using the new threshold
prediction = int(probability >= threshold)

# STEP 5: Display prediction results
print("\n===== AI PREDICTION RESULTS =====")

print("Failure probability:", round(probability * 100, 2), "%")
print("Decision threshold:", threshold * 100, "%")
print("AI prediction:", prediction)

# STEP 6: Interpret the prediction
if prediction == 1:
    print("\nWARNING: Potential machine failure detected!")
    print("Recommendation: Inspect machine operating conditions.")
else:
    print("\nMachine operating condition classified as normal.")

# STEP 7: Display risk assessment
print("\n===== MACHINE RISK ASSESSMENT =====")

if probability >= 0.70:
    print("Risk level: HIGH")
elif probability >= 0.30:
    print("Risk level: MODERATE")
else:
    print("Risk level: LOW")

print("\nPrediction completed successfully!")
