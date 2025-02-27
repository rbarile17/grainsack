"""Metrics for evaluating the performance of the LP-X methods."""

from sklearn.metrics import classification_report

from grainsack import EVALUATIONS_PATH, EXPLANATIONS_PATH
from grainsack.utils import read_json


def validation_metrics(kg, model, llm):
    """Compute the validation metrics for the LP-X methods."""
    method = "ground-truth"
    mode = "None"
    summarization = "None"

    gt = read_json(EXPLANATIONS_PATH / f"{method}_{mode}_{model}_{kg}_{summarization}.json")
    gt_fsv = [x["fsv"] for x in gt]

    fsv = read_json(EVALUATIONS_PATH / f"{method}_{mode}_{model}_{kg}_{summarization}_{llm}.json")
    fsv = [x["fsv"] for x in fsv]

    report = classification_report(gt_fsv, fsv, output_dict=True)

    return report
