from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\.externals\.loky\.backend\.context")

import model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GrocyGenie consumption model.")
    parser.add_argument("--data", type=Path, default=model.DATA_PATH, help="CSV training data path.")
    parser.add_argument(
        "--model-type",
        choices=["hist_gradient_boosting", "random_forest"],
        default="hist_gradient_boosting",
        help="Estimator to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = model.train_model(data_path=args.data, model_type=args.model_type)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
