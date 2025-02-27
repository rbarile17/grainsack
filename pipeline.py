import re
import subprocess
import time

import luigi
from grainsack import (
    EXPLANATIONS_PATH,
    KGES_PATH,
    KGS_PATH,
    LP_CONFIGS_PATH,
    PREDICTIONS_PATH,
)

from grainsack import format_fr200k


def run(script, args):
    submit_cmd = ["python", "-m", "grainsack_cli", script] + args
    print(submit_cmd)
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
            job_state = status_lines[0].split()[0]  # Example: COMPLETED, FAILED, etc.
        else:
            job_state = None

        print(f"Job {job_id} status: {job_state}")

        if job_state in ["COMPLETED"]:
            print(f"Job {job_id} completed successfully!")
            break
        elif job_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]:
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
        return (
            FormatFR200K()
            if self.kg == "FR200K"
            else map(luigi.LocalTarget, [KGS_PATH / "self.kg" / f"{f}.txt" for f in ["train", "valid", "test"]])
        )

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
    mode = luigi.Parameter()
    summarization = luigi.Parameter()

    def requires(self):
        return SelectPredictions(kg=self.kg, model=self.model)

    def output(self):
        return luigi.LocalTarget(EXPLANATIONS_PATH / f"{self.method}_{self.mode}_{self.model}_{self.kg}_{self.summarization}.txt")

    def run(self):
        run("explain")

class Dixit(luigi.Task):
    kg = luigi.Parameter()
    model = luigi.Parameter()
    method = luigi.Parameter()
    mode = luigi.Parameter()
    summarization = luigi.Parameter()

    def requires(self):
        return SelectPredictions(kg=self.kg, model=self.model)

    def output(self):
        return luigi.LocalTarget(EXPLANATIONS_PATH / f"{self.method}_{self.mode}_{self.model}_{self.kg}_{self.summarization}.txt")

    def run(self):
        run("dixit")


class MasterTask(luigi.WrapperTask):
    def requires(self):
        return [Explain(kg="FR200K", model="TransE")]


if __name__ == "__main__":
    luigi.run()
