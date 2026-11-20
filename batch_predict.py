
import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

# STEP 1: Load manufacturing dataset
dataset = fetch_ucirepo(id=601)

df = dataset.data.features.copy()
df["ActualFailure"] = dataset.data.targets[
    "Machine failure"
].to_numpy()

# STEP 2: Select 20 example observations
normal = df[df["ActualFailure"] == 0].sample(
    10, random_state=42
)

failed = df[df["ActualFailure"] == 1].sample(
    10, random_state=42
)

machines = pd.concat([normal, failed])
machines = machines.sample(
    frac=1, random_state=42
).reset_index(drop=True)

# Save input CSV
machines.to_csv("machine_readings.csv", index=False)

print("20 machine observations saved to CSV.")

# STEP 3: Read the CSV
machines = pd.read_csv("machine_readings.csv")

# STEP 4: Configure our AI API
url = "http://127.0.0.1:8000/predict"

results = []

# STEP 5: Process each observation
for index, row in machines.iterrows():

    machine_id = f"DEMO-{index + 1:03d}"

    payload = {
        "machine_id": machine_id,
        "air_temperature": float(row["Air temperature"]),
        "process_temperature": float(row["Process temperature"]),
        "rotational_speed": float(row["Rotational speed"]),
        "torque": float(row["Torque"]),
        "tool_wear": float(row["Tool wear"])
    }

    # Send measurements to the AI service
    response = requests.post(
        url, json=payload, timeout=10
    )

    response.raise_for_status()
    prediction = response.json()

    # Store results for analysis
    results.append({
        "MachineId": machine_id,
        "ActualFailure": int(row["ActualFailure"]),
        "PredictedFailure": prediction["failurePrediction"],
        "FailureProbability": prediction["failureProbability"],
        "RiskLevel": prediction["riskLevel"]
    })

# STEP 6: Convert results into a DataFrame
results_df = pd.DataFrame(results)

# STEP 7: Export prediction results
results_df.to_csv(
    "machine_predictions.csv", index=False
)

print("\n===== MACHINE HEALTH REPORT =====")
print(results_df.to_string(index=False))

# STEP 8: Summarise predictions
print("\n===== SUMMARY =====")

print("Total observations:", len(results_df))

print("Failures flagged:", (
    results_df["PredictedFailure"] == 1
).sum())

print("Actual failures:", (
    results_df["ActualFailure"] == 1
).sum())

print("Missed failures:", (
    (results_df["ActualFailure"] == 1) &
    (results_df["PredictedFailure"] == 0)
).sum())

print("\nResults saved to machine_predictions.csv")
