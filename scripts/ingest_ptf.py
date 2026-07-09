import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.epias.client import EpiasCredentialsError
from data_pipeline.epias.ptf_ingestion import PtfIngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest hourly EPİAŞ PTF data into PostgreSQL."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=date.fromisoformat,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=date.fromisoformat,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Number of calendar days per EPİAŞ request (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = PtfIngestionService()
    try:
        summary = service.ingest_ptf_range(
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_days=args.chunk_days,
        )
    except (EpiasCredentialsError, ValueError) as exc:
        print(f"PTF ingestion could not start: {exc}", file=sys.stderr)
        return 2
    finally:
        service.client.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
