"""This module contains constants used throughout the grainsack package."""

from pathlib import Path

MAX_PROCESSES = 8

EXPERIMENTS_PATH = Path("experiments")

LP_CONFIGS_PATH = EXPERIMENTS_PATH / "kge_configs"
LOGS_PATH = EXPERIMENTS_PATH / "logs"
KGES_PATH = EXPERIMENTS_PATH / "kge_models"
EXPLANATIONS_PATH = EXPERIMENTS_PATH / "explanations"
PREDICTIONS_PATH = EXPERIMENTS_PATH / "predictions"
SELECTED_PREDICTIONS_PATH = EXPERIMENTS_PATH / "selected_predictions"
EVALUATIONS_PATH = EXPERIMENTS_PATH / "evaluations"
METRICS_PATH = EXPERIMENTS_PATH / "metrics"
KGS_PATH = Path("kgs")

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