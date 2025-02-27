import re
import subprocess
import time
import json
import luigi
from grainsack import EXPLANATIONS_PATH, KGES_PATH, KGS_PATH, LP_CONFIGS_PATH, PREDICTIONS_PATH, DIXIT_PATH, GROUND_TRUTH

from grainsack.format_ground_truth import format_fr200k

from grainsack.utils import read_json

from grainsack.metrics import validation_metrics


def run(script, args):
    submit_cmd = ["python", "-m", "grainsack.cli", script] + args
    subprocess.run(submit_cmd)


def run_slurm(script):
    submit_cmd = ["sbatch", f"slurm/{script}.sh"]
    output = subprocess.check_output(submit_cmd).decode("utf-8")
    match = re.search(r"Submitted batch job (\d+)", output)
    job_id = match.group(1)
    while True:
        status_cmd = ["sacct", "-j", job_id, "--format=State", "--noheader"]
        status_output = subprocess.check_output(status_cmd).decode("utf-8").strip()

        status_lines = [line.strip() for line in status_output.splitlines() if line.strip()]
        if status_lines:
            job_state = status_lines[0].split()[0]
        else:
            job_state = None

        print(f"Job {job_id} status: {job_state}")

        if job_state in ["COMPLETED"]:
            print(f"Job {job_id} completed successfully!")
            break
        if job_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]:
            raise RuntimeError(f"Job {job_id} failed with state {job_state}")
        else:
            time.sleep(10)


class FormatFR200K(luigi.Task):
    def output(self):
        return map(luigi.LocalTarget, [KGS_PATH / "FR200K" / f"{f}.txt" for f in ["train", "valid", "test"]])

    def requires(self):
        return luigi.LocalTarget(KGS_PATH / "FR200K" / "FR200K.npz")

    def run(self):
        format_fr200k()


class Tune(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()

    def requires(self):
        if self.kg == "FR200K":
            return FormatFR200K()
        return map(luigi.LocalTarget, [KGS_PATH / "self.kg" / f"{f}.txt" for f in ["train", "valid", "test"]])

    def params_as_args(self):
        return ["--kg", self.kg, "--model", self.model]

    def output(self):
        return luigi.LocalTarget(LP_CONFIGS_PATH / f"{self.model}_{self.kg}.json")

    def run(self):
        run("tune", self.params_as_args())


class Train(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()

    def requires(self):
        return Tune(kg=self.kg, model=self.model)

    def output(self):
        return luigi.LocalTarget(KGES_PATH / f"{self.model}_{self.kg}.pt")

    def params_as_args(self):
        return ["--kg", self.kg, "--model", self.model]

    def run(self):
        run("train", self.params_as_args())


class Predict(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()

    def requires(self):
        return Train(kg=self.kg, model=self.model)

    def output(self):
        return luigi.LocalTarget(PREDICTIONS_PATH / f"{self.model}_{self.kg}.csv")

    def params_as_args(self):
        return ["--kg", self.kg, "--model", self.model]

    def run(self):
        run("predict", self.params_as_args())


class SelectPredictions(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()

    def requires(self):
        return Predict(kg=self.kg, model=self.model)

    def output(self):
        return luigi.LocalTarget(PREDICTIONS_PATH / f"{self.model}_{self.kg}2.csv")

    def params_as_args(self):
        return ["--kg", self.kg, "--model", self.model]

    def run(self):
        run("select-predictions", self.params_as_args())


class Explain(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()
    method = luigi.Parameter()
    explanation_kwargs = luigi.Parameter()

    def requires(self):
        return SelectPredictions(model=self.model, kg=self.kg)

    def params_as_args(self):
        args = ["--model", self.model, "--kg", self.kg, "--method", self.method, "--explanation_kwargs", self.explanation_kwargs]
        return args

    def output(self):
        explanation_kwargs = json.loads(self.explanation_kwargs)
        kwargs_str = "_".join(str(v) if v is not None else "null" for v in explanation_kwargs.values())
        file = EXPLANATIONS_PATH / f"{self.model}_{self.kg}_{self.method}_{kwargs_str}.json"
        return luigi.LocalTarget(file)

    def run(self):
        run("explain", self.params_as_args())


class Dixit(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()
    method = luigi.Parameter()
    explanation_kwargs = luigi.Parameter()
    llm = luigi.Parameter()

    def requires(self):
        return Explain(model=self.model, kg=self.kg, method=self.method, explanation_kwargs=self.explanation_kwargs)

    def output(self):
        explanation_kwargs = json.loads(self.explanation_kwargs)
        kwargs_str = "_".join(str(v) if v is not None else "null" for v in explanation_kwargs.values())
        file = DIXIT_PATH / f"{self.model}_{self.kg}_{self.method}_{self.llm}_{kwargs_str}.json"
        return luigi.LocalTarget(file)

    def params_as_args(self):
        args = [
            "--model",
            self.model,
            "--kg",
            self.kg,
            "--method",
            self.method,
            "--explanation_kwargs",
            self.explanation_kwargs,
            "--llm",
            self.llm,
        ]
        return args

    def run(self):
        run("dixit", self.params_as_args())


class ComputeComparisonMetrics(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()
    method = luigi.Parameter()
    explanation_kwargs = luigi.Parameter()
    llm = luigi.Parameter()

    def requires(self):
        return Dixit(kg=self.kg, model=self.model, method=self.method, explanation_kwargs=self.explanation_kwargs, llm=self.llm)

    def output(self):
        explanation_kwargs = json.loads(self.explanation_kwargs)
        kwargs_str = "_".join(str(v) if v is not None else "null" for v in explanation_kwargs.values())
        file = DIXIT_PATH / f"{self.model}_{self.kg}_{self.method}_{self.llm}_{kwargs_str}_metrics.json"
        return luigi.LocalTarget(file)

    def run(self):
        path = DIXIT_PATH / f"{self.method}_{self.mode}_{self.model}_{self.kg}_{self.summarization}_{self.llm}.json"
        labeled_explanations = read_json(path)
        fsv = [x["fsv"] for x in labeled_explanations]

        return sum(fsv) / len(fsv)


class ComputeValidationMetrics(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()
    llm = luigi.Parameter()

    def requires(self):
        return Dixit(
            kg=self.kg,
            model=self.model,
            method=GROUND_TRUTH,
            explanation_kwargs='{"mode": null, "summarization": null}',
            llm=self.llm,
        )

    def output(self):
        file = DIXIT_PATH / f"{GROUND_TRUTH}_{self.model}_{self.kg}_{self.llm}_metrics.json"
        return luigi.LocalTarget(file)

    def run(self):
        return validation_metrics(kg=self.kg, model=self.model, llm=self.llm)


class Comparison(luigi.WrapperTask):
    def requires(self):
        return [ComputeComparisonMetrics(kg="FR200K", model="TransE", llm="")]


class Validation(luigi.WrapperTask):
    def requires(self):
        return [ComputeValidationMetrics(kg="FR200K", model="TransE", llm="")]


if __name__ == "__main__":
    luigi.run()
