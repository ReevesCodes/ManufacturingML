# ManufacturingML

ManufacturingML simulates equipment telemetry, trains machine-specific Random Forest classifiers, and serves failure-risk predictions through FastAPI. The repository includes trained models, example requests, and evaluation reports for 12 machine profiles. The evaluation scores describe performance on synthetic data. Field performance has yet to be established.

## Run the API

From the repository root, create a Python virtual environment and install the dependencies. The scikit-learn pin matches the environment used to load the bundled models.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r FabCore_CLN101_Training/requirements.txt 'scikit-learn==1.6.1' fastapi uvicorn
python -m uvicorn api:app
```

Open the interactive API docs at `http://127.0.0.1:8000/docs`, or send one of the bundled requests from a second terminal.

```bash
curl --fail-with-body http://127.0.0.1:8000/models/PACK-101/schema
curl --fail-with-body http://127.0.0.1:8000/predict/machine \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/PACK-101/example_request.json
```

## Training and evaluation

- [Fleet models](FLEET_MODELS.md) lists the 12 profiles, evaluation scores, and training commands.
- [Training pipeline](PIPELINE.md) explains how the simulator labels readings and how training splits data by maintenance cycle.
- [CLN-101 walkthrough](FabCore_CLN101_Training/README.md) covers the original single-machine workflow.
- [API integration](MACHINE_AI_INTEGRATION.md) documents prediction payloads and model loading.
