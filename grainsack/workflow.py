import hashlib
import json
import subprocess

import luigi
import pandas as pd

from grainsack import (
    EVALUATIONS_PATH,
    EXPLANATIONS_PATH,
    GROUND_TRUTH,
    KGES_PATH,
    KGS_PATH,
    LP_CONFIGS_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    SELECTED_PREDICTIONS_PATH,
)
from grainsack.format_ground_truth import format_fr200k
from grainsack.utils import read_json, write_json
from sklearn.metrics import classification_report


def hash_json_string(*json_strings, length=8):
    parsed = [json.loads(s) for s in json_strings]
    serialized = json.dumps(parsed, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:length]


def run(script, args):
    """
    Run a script with the given arguments.
    Args:
        script: The name of the script to run.
        args: A list of arguments to pass to the script.
    """
    submit_cmd = ["python", "-m", "grainsack.cli", script] + args
    subprocess.run(submit_cmd, check=True)


class FormatFR200K(luigi.Task):
    """Format the FR200K dataset."""

    def output(self):
        """Output files for the formatted FR200K dataset."""
        return map(luigi.LocalTarget, [KGS_PATH / "FR200K" / f"{f}.txt" for f in ["train", "valid", "test"]])

    def requires(self):
        """Input files for the FR200K dataset."""
        return luigi.LocalTarget(KGS_PATH / "FR200K" / "FR200K.npz")

    def run(self):
        """Run the formatting process."""
        format_fr200k()


class Tune(luigi.Task):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"

    def requires(self):
        """Declare dependencies."""
        if self.kg_name == "FR200K":
            return FormatFR200K()
        return []
        # return map(luigi.LocalTarget, [KGS_PATH / self.kg_name / f"{f}.txt" for f in ["train", "valid", "test"]])

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_args(self):
        """Convert parameters to command line arguments."""
        return ["--kg_name", self.kg_name, "--kge_model_name", self.kge_model_name, "--output_path", self.output_path]

    def run(self):
        """Execute the task."""
        run("tune", self.params_as_args())


class Train(luigi.Task):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.output_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"

    def requires(self):
        """Declare dependencies."""
        return Tune(kg_name=self.kg_name, kge_model_name=self.kge_model_name)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_args(self):
        """Convert parameters to command line arguments."""
        return ["--kg_name", self.kg_name, "--kge_config_path", self.kge_config_path, "--output_path", self.output_path]

    def run(self):
        """Execute the task."""
        run("train", self.params_as_args())


class Rank(luigi.Task):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kge_model_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"
        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.output_path = PREDICTIONS_PATH / f"{self.kg_name}_{self.kge_model_name}.csv"

    def requires(self):
        """Declare dependencies."""
        return Train(kg_name=self.kg_name, kge_model_name=self.kge_model_name)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_args(self):
        """Convert parameters to command line arguments."""
        return [
            "--kg_name",
            self.kg_name,
            "--kge_model_path",
            self.kge_model_path,
            "--kge_config_path",
            self.kge_config_path,
            "--output_path",
            self.output_path,
        ]

    def run(self):
        """Execute the task."""
        run("rank", self.params_as_args())


class SelectPredictions(luigi.Task):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.predictions_path = PREDICTIONS_PATH / f"{self.kg_name}_{self.kge_model_name}.csv"
        self.output_path = SELECTED_PREDICTIONS_PATH / f"{self.kg_name}_{self.kge_model_name}.csv"

    def requires(self):
        """Declare dependencies."""
        return Rank(kg_name=self.kg_name, kge_model_name=self.kge_model_name)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_args(self):
        """Convert parameters to command line arguments."""
        return ["--predictions_path", self.predictions_path, "--output_path", self.output_path]

    def run(self):
        """Execute the task."""
        run("select-predictions", self.params_as_args())


class Explain(luigi.Task):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()
    lpx_config = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lpx_config_hash = hash_json_string(self.lpx_config)

        self.predictions_path = SELECTED_PREDICTIONS_PATH / f"{self.kg_name}_{self.kge_model_name}.csv"
        self.kge_model_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"
        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.output_path = EXPLANATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config_hash}.json"

    def requires(self):
        """Declare dependencies."""
        return SelectPredictions(kg_name=self.kg_name, kge_model_name=self.kge_model_name)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_args(self):
        """Convert parameters to command line arguments."""

        args = [
            "--predictions_path",
            self.predictions_path,
            "--kg_name",
            self.kg_name,
            "--kge_model_path",
            self.kge_model_path,
            "--kge_config_path",
            self.kge_config_path,
            "--lpx_config",
            self.lpx_config,
            "--output_path",
            self.output_path,
        ]
        return args

    def run(self):
        """Execute the task."""
        run("explain", self.params_as_args())


class Evaluate(luigi.Task):
    """Run the DIXIT task."""

    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()
    lpx_config = luigi.Parameter()
    eval_config = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lpx_config_hash = hash_json_string(self.lpx_config)
        lpx_and_eval_config_hash = hash_json_string(self.lpx_config, self.eval_config)

        self.kge_model_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"
        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.explanations_path = EXPLANATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config_hash}.json"
        self.output_path = EVALUATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_and_eval_config_hash}.json"

    def requires(self):
        """Declare dependencies."""
        return Explain(kg_name=self.kg_name, kge_model_name=self.kge_model_name, lpx_config=self.lpx_config)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_args(self):
        """Convert parameters to command line arguments."""
        args = [
            "--explanations_path",
            self.explanations_path,
            "--kg_name",
            self.kg_name,
            "--kge_model_path",
            self.kge_model_path,
            "--kge_config_path",
            self.kge_config_path,
            "--eval_config",
            self.eval_config,
            "--output_path",
            self.output_path,
        ]
        return args

    def run(self):
        """Execute the task."""
        run("evaluate", self.params_as_args())


class Metrics(luigi.Task):
    """Compute the comparison metrics."""

    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()
    lpx_config = luigi.Parameter()
    eval_config = luigi.Parameter()
    metric_name = luigi.Parameter()

    def requires(self):
        """Declare dependencies"""
        return Evaluate(
            kg_name=self.kg_name, kge_model_name=self.kge_model_name, lpx_config=self.lpx_config, eval_config=self.eval_config
        )

    def output(self):
        """Declare output file."""
        config_hash = hash_json_string(self.lpx_config, self.eval_config)
        file = METRICS_PATH / f"{self.kg_name}_{self.kge_model_name}_{config_hash}.json"
        return luigi.LocalTarget(file)

    def run(self):
        """Execute the task."""
        config_hash = hash_json_string(self.lpx_config, self.eval_config)
        file = EVALUATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{config_hash}.json"
        evaluated_explanations = read_json(file)

        if self.metric_name == "validation_metrics":
            print("Validation metrics")
            lpx_config_hash = hash_json_string(self.lpx_config)
            gt = read_json(EXPLANATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config_hash}.json")
            gt_fsv = [x["fsv"] for x in gt]
            fsv = [x["fsv"] for x in evaluated_explanations]

            metrics = classification_report(gt_fsv, fsv, output_dict=True)
        elif self.metric_name == "comparison_metrics":
            fsv = [x["fsv"] for x in evaluated_explanations]

            metrics = {
                "average_fsv": sum(fsv) / len(fsv),
                "fsv_distribution": {
                    "1": fsv.count(1),
                    "0": fsv.count(0),
                    "-1": fsv.count(-1),
                },
            }

        output = METRICS_PATH / f"{self.kg_name}_{self.kge_model_name}_{config_hash}.json"

        write_json(metrics, output)


class Comparison(luigi.WrapperTask):
    """Represent the DAG of a setup for comparison experiments."""

    def requires(self):
        """Instantiate the DAG from the experimental setup."""

        df = pd.read_csv("comparison_setup.csv", sep=";")
        return [Metrics(**row, metric_name="comparison_metrics") for row in df.to_dict(orient="records")]


class Validation(luigi.WrapperTask):
    """Represent the DAG of a setup for validation experiments."""

    def requires(self):
        """Instantiate the DAG from the experimental setup."""

        df = pd.read_csv("validation_setup.csv", sep=";")
        return [Metrics(**row, lpx_config="{\"method\": \"ground-truth\"}", metric_name="validation_metrics") for row in df.to_dict(orient="records")]
