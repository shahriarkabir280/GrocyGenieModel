# GrocyGenie Model API

GrocyGenie predicts when grocery stock is likely to run out based on product type,
household size, location, season, household event, quantity, and purchase date.

The project is structured as a production-style machine learning service:

- FastAPI REST API for mobile or backend integration
- scikit-learn training pipeline with saved artifacts
- repeatable train/evaluate scripts
- model metrics saved to `artifacts/metrics.json`
- optional Supabase integration for user stock records and feedback

## Project Structure

```text
.
├── app.py                 # FastAPI app
├── model.py               # Feature engineering, training, prediction, feedback retraining
├── supabase_client.py     # Lazy Supabase client
├── initial_data.csv       # Training dataset
├── scripts/
│   ├── train.py           # Train and save model artifacts
│   └── evaluate.py        # Evaluate saved model on a held-out split
├── requirements.txt
└── runtime.txt
```

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.10` is not available locally, use any Python version supported by the
dependencies. The deployed runtime is pinned in `runtime.txt`.

## Train

```bash
python scripts/train.py
```

This creates:

- `artifacts/consumption_model.joblib`
- `artifacts/metrics.json`

You can also compare estimators:

```bash
python scripts/train.py --model-type random_forest
python scripts/train.py --model-type hist_gradient_boosting
```

## Evaluate

```bash
python scripts/evaluate.py
```

The evaluation reports:

- MAE: average daily-consumption error
- RMSE: error with larger mistakes weighted more heavily
- R2: explained variance
- MAPE: percentage error
- per-product MAE and MAPE

These metrics make the project easier to discuss in a resume or interview because
the model quality is measurable and reproducible.

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

## Run API

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Example Prediction

```bash
curl -X POST http://127.0.0.1:8000/re-predict \
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

## Supabase Integration

The `/stock/add`, `/feedback`, and `/retrain` endpoints use Supabase.

Create a `.env` file:

```bash
SUPABASE_URL=your-project-url
SUPABASE_KEY=your-service-or-anon-key
```

Local training and evaluation do not require Supabase credentials.

## Resume Highlights

This project demonstrates:

- end-to-end ML lifecycle: training, persistence, evaluation, API serving
- feature engineering for categorical, household, date, and domain-specific signals
- input validation and error handling in FastAPI
- feedback loop for improving predictions from real user outcomes
- reproducible metrics suitable for model comparison

## Important Modeling Note

The included dataset appears synthetic or semi-synthetic. That is useful for
building the system, but resume claims should focus on engineering quality and
measured offline performance unless the model is later validated with real user
consumption data.
