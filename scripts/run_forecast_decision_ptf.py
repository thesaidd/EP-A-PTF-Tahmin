import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.forecast_decision_ptf import ForecastDecisionPtfService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PTF forecast decision layer.")
    parser.add_argument("--gpr-run-id", default=None)
    parser.add_argument("--model-version", default="forecast_decision_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = ForecastDecisionPtfService().run_decision_layer(
        gpr_run_id=args.gpr_run_id,
        model_version=args.model_version,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
