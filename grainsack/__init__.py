"""This module contains constants used throughout the grainsack package."""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('grainsack')

MAX_PROCESSES = 8

EXPERIMENTS_PATH = Path("/lustrehome/robertobarile/grainsack/experiments")

LP_CONFIGS_PATH = EXPERIMENTS_PATH / "kge_configs"
LOGS_PATH = EXPERIMENTS_PATH / "logs"
KGES_PATH = EXPERIMENTS_PATH / "kge_models"
EXPLANATIONS_PATH = EXPERIMENTS_PATH / "explanations"
PREDICTIONS_PATH = EXPERIMENTS_PATH / "predictions"
SELECTED_PREDICTIONS_PATH = EXPERIMENTS_PATH / "selected_predictions"
EVALUATIONS_PATH = EXPERIMENTS_PATH / "evaluations"
METRICS_PATH = EXPERIMENTS_PATH / "metrics"
KGS_PATH = Path("/lustrehome/robertobarile/grainsack/kgs")

FRUNI = "FRUNI"
FR200K = "FR200K"

CONVE = "ConvE"
COMPLEX = "ComplEx"
TRANSE = "TransE"

CRIAGE = "criage"
DATA_POISONING = "dp"
KELPIE = "kelpie"
KELPIEPP = "kelpie++"
IMAGINE = "imagine"

SIMULATION = "simulation"
