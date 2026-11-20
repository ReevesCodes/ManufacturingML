
import requests

# Address of our machine learning API
url = "http://127.0.0.1:8000/predict"

# Machine sensor measurements
machine_data = {
    "machine_id": "DEMO-001",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 25
}

# Send machine data to our AI service
response = requests.post(
    url,
    json=machine_data,
    timeout=10
)

# Check whether the request succeeded
response.raise_for_status()

# Convert the returned JSON into a Python dictionary
result = response.json()

print("\n===== AI MACHINE HEALTH REPORT =====")

print("Machine:", result["machineId"])
print("Failure prediction:", result["failurePrediction"])
print(
    "Failure probability:",
    round(result["failureProbability"] * 100, 2),
    "%"
)
print("Risk level:", result["riskLevel"])

if result["failurePrediction"] == 1:
    print("ALERT: Potential machine failure!")
else:
    print("Machine classified as normal.")
