import hashlib
import json
import re
import subprocess
import time

import luigi
import pandas as pd

from grainsack import (
    EVALUATIONS_PATH,
    EXPLANATIONS_PATH,
    KGES_PATH,
    LOGS_PATH,
    LP_CONFIGS_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    SELECTED_PREDICTIONS_PATH,
)
from grainsack.utils import read_json, write_json


def hash_json_string(*json_strings, length=8):
    """Generate a short hash from one or more JSON strings.
    
    Parses JSON strings, sorts keys for consistent ordering, and generates
    an MD5 hash for use in unique identifiers.
    
    Args:
        *json_strings (str): One or more JSON-formatted strings.
        length (int, optional): Length of the returned hash. Defaults to 8.
        
    Returns:
        str: Hexadecimal hash string of specified length.
    """
    parsed = [json.loads(s) for s in json_strings]
    serialized = json.dumps(parsed, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:length]


def run(script, args):
    """Execute a grainsack operation script directly with Python.
    
    Runs a script from the grainsack.operations module with the given arguments.
    
    Args:
        script (str): Name of the script/command to run (e.g., 'train', 'explain').
        args (list): Command-line arguments to pass to the script.
        
    Raises:
        subprocess.CalledProcessError: If the script execution fails.
    """
    submit_cmd = ["python", "-m", "grainsack.operations", script] + args
    subprocess.run(submit_cmd, check=True)


def run_slurm(script, args):
    """Submit and monitor a SLURM batch job.
    
    Submits a job to SLURM using sbatch, monitors its status, and blocks
    until completion or failure.
    
    Args:
        script (str): Name of the SLURM script in slurm/ directory (without .sh).
        args (list): Arguments to pass to the SLURM script.
        
    Raises:
        RuntimeError: If the job fails, is cancelled, times out, or node fails.
    """
    submit_cmd = ["sbatch", f"slurm/{script}.sh"] + args
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

        if job_state in ["COMPLETED"]:
            break
        if job_state in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]:
            raise RuntimeError(f"Job {job_id} failed with state {job_state}")
        time.sleep(10)


def run_kubernetes(script, params_dict):
    """Submit and monitor a Kubernetes job.
    
    Creates a Kubernetes job from a template, submits it, monitors its status,
    and blocks until completion or failure. Cleans up temporary files.
    
    Args:
        script (str): Name of the script (e.g., 'tune', 'train'), used to
            locate template in kubernetes/ directory.
        params_dict (dict): Parameter dictionary converted to command-line
            arguments for the container.
            
    Raises:
        RuntimeError: If the Kubernetes job fails.
    """
    import tempfile
    import yaml
    import os
    
    # Generate unique job name
    job_name = f"{script}-{hash_json_string(json.dumps(params_dict, sort_keys=True))}"
    
    # Read the template
    template_path = f"kubernetes/{script}.yml"
    with open(template_path, 'r') as f:
        job_spec = yaml.safe_load(f)
    
    # Update job spec with parameters
    job_spec['metadata']['name'] = job_name
    
    # Update command with parameters
    container = job_spec['spec']['template']['spec']['containers'][0]
    cmd_args = []
    for key, value in params_dict.items():
        cmd_args.extend([f"--{key}", str(value)])
    
    # Replace the command arguments
    container['command'] = ["python", "-u", "-m", f"grainsack.operations", script] + cmd_args
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(job_spec, f)
        temp_file = f.name
    
    try:
        # Create the job
        create_cmd = ["kubectl", "create", "-f", temp_file]
        subprocess.run(create_cmd, check=True)
        
        # Monitor the job
        while True:
            status_cmd = ["kubectl", "get", "job", job_name, "-o", "jsonpath={.status.conditions[?(@.type=='Complete')].status}"]
            try:
                complete_status = subprocess.check_output(status_cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            except subprocess.CalledProcessError:
                complete_status = ""
            
            status_cmd_failed = ["kubectl", "get", "job", job_name, "-o", "jsonpath={.status.conditions[?(@.type=='Failed')].status}"]
            try:
                failed_status = subprocess.check_output(status_cmd_failed, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            except subprocess.CalledProcessError:
                failed_status = ""
            
            if complete_status == "True":
                break
            if failed_status == "True":
                raise RuntimeError(f"Kubernetes job {job_name} failed")
            
            time.sleep(10)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


class BaseTask(luigi.Task):
    """Base Luigi task with common parameter conversion functionality.
    
    Provides standardized methods for converting task parameters to different
    formats needed by various execution backends (direct Python, SLURM, Kubernetes).
    Subclasses should implement params_as_dict() and optionally override other methods
    for custom parameter handling.
    """
    
    def params_as_dict(self):
        """Convert task parameters to a dictionary.
        
        Returns:
            dict: Parameter names and values.
            
        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError
    
    def params_as_args(self):
        """Convert parameters to command line arguments."""
        params = self.params_as_dict()
        args = []
        for key, value in params.items():
            args.extend([f"--{key}", str(value)])
        return args
    
    def slurm_params_as_args(self):
        """Convert parameters to SLURM-specific argument list.
        
        By default, returns parameter values without keys. Override in subclasses
        if SLURM scripts require different formatting (e.g., including log paths).
        
        Returns:
            list: Parameter values as a list.
        """
        return list(self.params_as_dict().values())
    
    def kubernetes_params_as_dict(self):
        """Convert parameters to Kubernetes-compatible dictionary.
        
        Converts all Path objects to strings for YAML serialization. Override
        if additional parameter transformations are needed.
        
        Returns:
            dict: Parameters with Path objects converted to strings.
        """
        return {k: str(v) for k, v in self.params_as_dict().items()}
    
    def run_with_executor(self, script_name):
        """Execute task using the appropriate backend executor.
        
        Determines the execution backend from the EXECUTOR environment variable
        and runs the task using either Kubernetes or SLURM.
        
        Args:
            script_name (str): Name of the operation script to execute.
        """
        
        import os
        if os.getenv("EXECUTOR") == "kubernetes":
            run_kubernetes(script_name, self.kubernetes_params_as_dict())
        else:
            run_slurm(script_name, self.slurm_params_as_args())


class Tune(BaseTask):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.log_path = LOGS_PATH / f"tune_{self.kg_name}_{self.kge_model_name}.log"

    def requires(self):
        """Declare dependencies."""
        return []

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_dict(self):
        """Convert parameters to dictionary."""
        return {
            "kg_name": self.kg_name,
            "kge_model_name": self.kge_model_name,
            "output_path": self.output_path
        }
    
    def slurm_params_as_args(self):
        """SLURM needs log_path as well."""
        return [self.kg_name, self.kge_model_name, self.output_path, self.log_path]

    def run(self):
        """Execute the task."""
        self.run_with_executor("tune")


class Train(BaseTask):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.output_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"
        self.log_path = LOGS_PATH / f"train_{self.kg_name}_{self.kge_model_name}.log"

    def requires(self):
        """Declare dependencies."""
        return Tune(kg_name=self.kg_name, kge_model_name=self.kge_model_name)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_dict(self):
        """Convert parameters to dictionary."""
        return {
            "kg_name": self.kg_name,
            "kge_config_path": self.kge_config_path,
            "output_path": self.output_path
        }
    
    def slurm_params_as_args(self):
        """SLURM needs log_path as well."""
        return [self.kg_name, self.kge_config_path, self.output_path, self.log_path]

    def run(self):
        """Execute the task."""
        self.run_with_executor("train")


class Rank(BaseTask):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kge_model_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"
        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.output_path = PREDICTIONS_PATH / f"{self.kg_name}_{self.kge_model_name}.csv"
        self.log_path = LOGS_PATH / f"rank_{self.kg_name}_{self.kge_model_name}.log"

    def requires(self):
        """Declare dependencies."""
        return Train(kg_name=self.kg_name, kge_model_name=self.kge_model_name)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_dict(self):
        """Convert parameters to dictionary."""
        return {
            "kg_name": self.kg_name,
            "kge_model_path": self.kge_model_path,
            "kge_config_path": self.kge_config_path,
            "output_path": self.output_path,
        }

    def slurm_params_as_args(self):
        """SLURM needs log_path as well."""
        return [self.kg_name, self.kge_model_path, self.kge_config_path, self.output_path, self.log_path]

    def run(self):
        """Execute the task."""
        self.run_with_executor("rank")


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


class Explain(BaseTask):
    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()
    lpx_config = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lpx_config = json.loads(self.lpx_config)
        self.predictions_path = SELECTED_PREDICTIONS_PATH / f"{self.kg_name}_{self.kge_model_name}.csv"
        self.kge_model_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"
        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.output_path = (
            EXPLANATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config['method']}_{lpx_config['summarization']}.json"
        )
        self.log_path = (
            LOGS_PATH / f"explain_{self.kg_name}_{self.kge_model_name}_{lpx_config['method']}_{lpx_config['summarization']}.log"
        )

    def requires(self):
        """Declare dependencies."""
        return SelectPredictions(kg_name=self.kg_name, kge_model_name=self.kge_model_name)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_dict(self):
        """Convert parameters to dictionary."""
        return {
            "predictions_path": self.predictions_path,
            "kg_name": self.kg_name,
            "kge_model_path": self.kge_model_path,
            "kge_config_path": self.kge_config_path,
            "lpx_config": self.lpx_config,
            "output_path": self.output_path,
        }

    def slurm_params_as_args(self):
        """SLURM needs log_path as well."""
        return [
            self.predictions_path,
            self.kg_name,
            self.kge_model_path,
            self.kge_config_path,
            self.lpx_config,
            self.output_path,
            self.log_path,
        ]

    def run(self):
        """Execute the task."""
        self.run_with_executor("explain")


class Evaluate(BaseTask):
    """Run the DIXIT task."""

    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()
    lpx_config = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lpx_config = json.loads(self.lpx_config)
        self.kge_model_path = KGES_PATH / f"{self.kg_name}_{self.kge_model_name}.pt"
        self.kge_config_path = LP_CONFIGS_PATH / f"{self.kg_name}_{self.kge_model_name}.json"
        self.explanations_path = (
            EXPLANATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config['method']}_{lpx_config['summarization']}.json"
        )
        self.output_path = (
            EVALUATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config['method']}_{lpx_config['summarization']}.json"
        )
        self.log_path = (
            LOGS_PATH / f"evaluate_{self.kg_name}_{self.kge_model_name}_{lpx_config['method']}_{lpx_config['summarization']}.log"
        )

    def requires(self):
        """Declare dependencies."""
        return Explain(kg_name=self.kg_name, kge_model_name=self.kge_model_name, lpx_config=self.lpx_config)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def params_as_dict(self):
        """Convert parameters to dictionary."""
        return {
            "explanations_path": self.explanations_path,
            "kg_name": self.kg_name,
            "output_path": self.output_path
        }

    def slurm_params_as_args(self):
        """SLURM needs additional paths."""
        return [
            self.explanations_path,
            self.kg_name,
            self.kge_model_path,
            self.kge_config_path,
            self.output_path,
            self.log_path
        ]

    def run(self):
        """Execute the task."""
        self.run_with_executor("evaluate")


class Metrics(luigi.Task):
    """Compute the comparison metrics."""

    kg_name = luigi.Parameter()
    kge_model_name = luigi.Parameter()
    lpx_config = luigi.Parameter()
    metric_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lpx_config = json.loads(self.lpx_config)

        self.evaluations_path = (
            EVALUATIONS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config['method']}_{lpx_config['summarization']}.json"
        )
        self.output_path = (
            METRICS_PATH / f"{self.kg_name}_{self.kge_model_name}_{lpx_config['method']}_{lpx_config['summarization']}.json"
        )

    def requires(self):
        """Declare dependencies"""
        return Evaluate(kg_name=self.kg_name, kge_model_name=self.kge_model_name, lpx_config=self.lpx_config)

    def output(self):
        """Declare output file."""
        return luigi.LocalTarget(self.output_path)

    def run(self):
        """Execute the task."""
        evaluated_explanations = read_json(self.evaluations_path)

        fsv = [x["fsv"] for x in evaluated_explanations]

        metrics = {
            "average_fsv": sum(fsv) / len(fsv),
            "fsv_distribution": {"1": fsv.count(1), "0": fsv.count(0), "-1": fsv.count(-1)},
        }

        write_json(metrics, self.output_path)


class Comparison(luigi.WrapperTask):
    """Represent the DAG of a setup for comparison experiments."""

    def requires(self):
        """Instantiate the DAG from the experimental setup."""

        df = pd.read_csv("comparison_setup.csv", sep=";")
        return [Metrics(**row, metric_name="comparison_metrics") for row in df.to_dict(orient="records")]
