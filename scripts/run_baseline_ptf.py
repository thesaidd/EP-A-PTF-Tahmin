import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.models.baseline_ptf import BaselinePtfService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and store hourly PTF baseline forecasts."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="Inclusive evaluation start date (default: 2024-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Inclusive evaluation end date (default: latest feature timestamp).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = BaselinePtfService()
    try:
        summary = service.run_baseline_evaluation(
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except ValueError as exc:
        print(f"Baseline evaluation could not start: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

