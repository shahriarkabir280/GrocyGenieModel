# GrocyGenie Model API

GrocyGenie is a machine learning API that predicts when a grocery item is likely
to run out. It uses product details, household size, region, season, event type,
quantity, and purchase date to estimate daily consumption and calculate a
predicted depletion date.

The project is built as a portfolio-ready ML service with reproducible training,
offline evaluation, saved model artifacts, FastAPI serving, and an optional
Supabase feedback loop.

## Live Demo

The API is deployed on Hugging Face Spaces:

- Hugging Face Space: https://huggingface.co/spaces/shahriar031/GrocyGenie
- Live API: https://shahriar031-grocygenie.hf.space
- API Docs: https://shahriar031-grocygenie.hf.space/docs
- Health Check: https://shahriar031-grocygenie.hf.space/health
- Model Info: https://shahriar031-grocygenie.hf.space/model/info
- OpenAPI Schema: https://shahriar031-grocygenie.hf.space/openapi.json

Note: the hosted demo runs on Hugging Face free CPU hardware, so it may take a
short time to wake up after inactivity.

## Features

- Predicts grocery depletion dates from household and stock context
- Serves predictions through a FastAPI REST API
- Provides reproducible training and evaluation scripts
- Saves model metrics to `artifacts/metrics.json`
- Supports per-product performance reporting
- Includes optional Supabase endpoints for stock persistence, feedback, and retraining
- Can run locally without Supabase credentials for model training, evaluation, and prediction

## Model Performance

Current performance on a held-out 20% test split from `initial_data.csv`:

```text
MAE: 0.0435 kg/day
RMSE: 0.0661 kg/day
R2: 0.9892
MAPE: 6.29%
Median Absolute Error: 0.0249 kg/day
```

Interpretation: the model's daily consumption predictions are off by about
`43.5g/day` on average, with an average percentage error of `6.29%` on the
current dataset.

## Tech Stack

- Python
- FastAPI
- scikit-learn
- pandas
- NumPy
- joblib
- Supabase, optional
- Hugging Face Spaces for deployment

## Project Structure

```text
.
├── app.py                 # FastAPI application and API routes
├── model.py               # Feature engineering, training, prediction, feedback retraining
├── supabase_client.py     # Lazy Supabase client for database-backed endpoints
├── initial_data.csv       # Training dataset
├── artifacts/
│   └── metrics.json       # Saved evaluation metadata
├── scripts/
│   ├── train.py           # Train and save model artifacts
│   └── evaluate.py        # Evaluate the saved model on a held-out split
├── requirements.txt
└── runtime.txt
```

## Local Setup

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.10` is not available locally, use a Python version supported by the
dependencies. The deployment runtime is pinned in `runtime.txt`.

## Train the Model

```bash
python scripts/train.py
```

This creates:

```text
artifacts/consumption_model.joblib
artifacts/metrics.json
```

You can also compare supported estimators:

```bash
python scripts/train.py --model-type hist_gradient_boosting
python scripts/train.py --model-type random_forest
```

## Evaluate the Model

```bash
python scripts/evaluate.py
```

The evaluation reports MAE, RMSE, R2, MAPE, median absolute error, and
per-product MAE/MAPE.

## Run the API Locally

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open the interactive docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

Public/local prediction endpoints:

- `GET /` - root status message
- `GET /health` - service health check
- `GET /model/info` - trained model metadata and metrics
- `POST /re-predict` - calculate a depletion date without writing to the database

Supabase-backed endpoints:

- `POST /stock/add` - add stock record and save predicted finish date
- `POST /feedback` - record the actual finish date for a stock item
- `POST /retrain` - retrain with verified user feedback

## Example Prediction Request

Live endpoint:

```bash
curl -X POST https://shahriar031-grocygenie.hf.space/re-predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "rice",
    "quantity": 5,
    "unit": "kg",
    "region": "rural",
    "season": "summer",
    "event": "normal",
    "family": {
      "adult_male": 2,
      "adult_female": 1,
      "child": 2
    }
  }'
```

Example response:

```json
{
  "predicted_finish_date": "2026-09-27"
}
```

## Supabase Configuration

Supabase is optional. You do not need it to train, evaluate, run `/health`,
inspect `/model/info`, or use `/re-predict`.

Supabase is only required for endpoints that persist or use user stock records:

```text
POST /stock/add
POST /feedback
POST /retrain
```

To enable those endpoints locally, create a `.env` file:

```bash
SUPABASE_URL=your-project-url
SUPABASE_KEY=your-key
```

Do not commit `.env` files or private credentials.

## Deployment

The project is deployed on Hugging Face Spaces as a Docker/FastAPI service.

The server command used by the Docker deployment is:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

For other hosting platforms, use the platform-provided port when available:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Technical Highlights

- End-to-end ML lifecycle: data loading, feature engineering, training, evaluation, artifact saving, and API serving
- Reproducible model evaluation with a held-out test split
- Domain features such as family size, region, season, event, date features, and base consumption estimates
- Clean FastAPI request validation with typed inputs
- Optional database-backed feedback loop for future personalization
- Public deployment with live API documentation

## Modeling Note

The included dataset appears synthetic or semi-synthetic. It is useful for
building and demonstrating the ML service, but real-world performance should be
validated with actual user consumption data before making production claims.

