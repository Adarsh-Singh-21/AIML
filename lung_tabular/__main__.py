import argparse
import json
from pathlib import Path

from . import DATA, REPORTS, ARTIFACTS


def main():
    parser = argparse.ArgumentParser(description="Numerical lung cancer dataset classification — no images")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download-data")
    sub.add_parser("prepare-data")
    training = sub.add_parser("train")
    training.add_argument("--run-name", default="numerical_v1")
    prediction = sub.add_parser("predict")
    prediction.add_argument("--csv", type=Path, required=True)
    prediction.add_argument("--encoding", choices=["numeric", "survey"], default="numeric")
    prediction.add_argument("--output", type=Path)
    sub.add_parser("doctor")
    args = parser.parse_args()
    try:
        if args.command == "download-data":
            from .data import download_data
            result = download_data()
        elif args.command == "prepare-data":
            from .data import prepare_data
            result = prepare_data()
        elif args.command == "train":
            from .model import train_run
            result = train_run(args.run_name)
        elif args.command == "doctor":
            import sklearn, sys
            from . import FEATURES
            result = {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "device": "CPU", "features": len(FEATURES)}
        else:
            import pandas as pd
            from .model import resolve_run, load_model, predict_records
            run, _ = resolve_run()
            model, metadata = load_model(run)
            result_frame = predict_records(pd.read_csv(args.csv), model, metadata, args.encoding)
            if args.output:
                result_frame.to_csv(args.output, index=False)
            result = {"predictions": result_frame.to_dict(orient="records"),
                      "interpretation": "Dataset-label scores, not individual clinical cancer probabilities."}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
