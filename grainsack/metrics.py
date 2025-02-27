"""Metrics for evaluating the performance of the LP-X methods."""

from grainsack import EXPLANATIONS_PATH, DIXIT_PATH

from grainsack.utils import read_json

from sklearn.metrics import classification_report

def validation_metrics(kg, model, llm):
    method = "ground-truth"
    mode = "None"
    summarization = "None"

    gt = read_json(EXPLANATIONS_PATH / f"{method}_{mode}_{model}_{kg}_{summarization}.json")
    gt_fsv = [x["fsv"] for x in gt]

    fsv = read_json(DIXIT_PATH / f"{method}_{mode}_{model}_{kg}_{summarization}_{llm}.json")
    fsv = [x["fsv"] for x in fsv]

    report = classification_report(gt_fsv, fsv, output_dict=True)
