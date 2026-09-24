from __future__ import annotations

import argparse
import json
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\.externals\.loky\.backend\.context")

import joblib
from sklearn.model_selection import train_test_split

import model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved GrocyGenie model artifact.")
    parser.add_argument("--data", type=Path, default=model.DATA_PATH, help="CSV evaluation data path.")
    parser.add_argument("--model", type=Path, default=model.MODEL_PATH, help="Saved model artifact path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model}. Run scripts/train.py first.")

    artifact = joblib.load(args.model)
    pipeline = artifact["pipeline"]
    df = model.load_training_data(args.data)
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["product"],
    )
    metrics = model.evaluate_pipeline(pipeline, test_df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
