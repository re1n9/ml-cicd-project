# Iris Prediction API

A small machine-learning inference API. It classifies an iris flower from four measurements and exposes that model over HTTP.

The same repository is set up so a push to `main` runs tests and a Docker build on GitHub Actions. Render rebuilds the service from this Dockerfile only after those checks pass.

## API

`GET /health` returns `{"status": "ok"}`.

`POST /predict` accepts:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

and returns:

```json
{
  "predicted_class": "setosa"
}
```

Each measurement must be a number from 0 to 10. Missing or out-of-range fields return HTTP 422.

Interactive docs are at `/docs`.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

On macOS or Linux, activate the virtual environment with `source .venv/bin/activate`.

Open `http://localhost:8000/docs`.

`scripts/train_model.py` fits a logistic regression model on scikit-learn's iris dataset and writes `models/iris_model.joblib`. That file is committed so the service and CI load the same model instead of training again.

## Tests

```bash
python -m pytest tests/
```

The tests check the health endpoint, a known setosa sample, a missing field, and a value outside 0–10.

## Docker

```bash
docker build -t ml-prediction-api .
docker run --rm -p 8000:8000 ml-prediction-api
```

The process listens on `0.0.0.0` and on the `PORT` environment variable, which defaults to 8000. Render sets `PORT` for you (usually `10000`).

## Continuous integration

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`. It installs dependencies, runs pytest, then builds the Docker image. The image built in Actions is only a check that the Dockerfile works. Render builds its own image from the same Dockerfile at deploy time.

## Deploy on Render

1. Push this repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), choose **New** → **Blueprint**.
3. Connect the GitHub repository. Render reads `render.yaml`.
4. Confirm the web service uses the Docker runtime, the free plan, branch `main`, and **Auto-Deploy: After CI Checks Pass** (`autoDeployTrigger: checksPass`).
5. Apply the Blueprint and wait for the first deploy.

After CI checks pass, Render deploys the new commit. If a check fails, or if a commit has no checks, Render does not deploy that commit. The free instance sleeps when idle, so the first request after a quiet period can take longer while it wakes up.

Call the public URL:

```bash
curl https://<your-service>.onrender.com/health
curl -X POST https://<your-service>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```
