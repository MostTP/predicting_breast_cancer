#!/usr/bin/env python3
"""
Orchestrator: trains S-learner and T-learners on combined cohorts.

Usage:
    python train.py --metabric Breast_Cancer_METABRIC.csv
    python train.py --metabric Breast_Cancer_METABRIC.csv --tcga data/tcga/tcga.csv --external data/external/yau.csv
"""

import argparse
import logging
import sys
from pathlib import Path

from ml.data_utils import load_all_data
from ml.train_s_learner import train as train_s
from ml.train_t_learner import train as train_t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train breast cancer treatment models")
    parser.add_argument("--metabric", required=True, help="Path to METABRIC CSV")
    parser.add_argument("--tcga", default=None, help="Path to TCGA CSV (optional)")
    parser.add_argument("--external", default=None, help="Path to external test CSV (optional)")
    parser.add_argument("--skip-t-learner", action="store_true", help="Skip T-learner training")
    args = parser.parse_args()

    metabric_path = Path(args.metabric)
    tcga_path = Path(args.tcga) if args.tcga else None
    external_path = Path(args.external) if args.external else None

    logger.info("=" * 70)
    logger.info("MULTI-COHORT TRAINING PIPELINE")
    logger.info("=" * 70)

    # Load & harmonize
    train_df, val_df, test_df = load_all_data(
        metabric_path=metabric_path,
        tcga_path=tcga_path,
        external_path=external_path,
    )

    logger.info(f"Combined training set: {len(train_df)} rows")
    logger.info(f"Validation set: {len(val_df)} rows")
    if test_df is not None:
        logger.info(f"External test set: {len(test_df)} rows")

    # Train S-learner (primary model)
    train_s(train_df, val_df, test_df)

    # Train T-learners (optional, requires sufficient per-combo samples)
    if not args.skip_t_learner:
        train_t(train_df)

    logger.info("=" * 70)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("Restart your API to load the new models.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()