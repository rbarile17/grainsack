#!/usr/bin/env python3
"""
Check the completion status of experiments defined in comparison_setup.csv
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Define paths
BASE_PATH = Path(__file__).parent
COMPARISON_SETUP = BASE_PATH / "comparison_setup.csv"
EXPERIMENTS_PATH = BASE_PATH / "experiments"
EXPLANATIONS_PATH = EXPERIMENTS_PATH / "explanations"
PREDICTIONS_PATH = EXPERIMENTS_PATH / "predictions"
SELECTED_PREDICTIONS_PATH = EXPERIMENTS_PATH / "selected_predictions"
KGE_CONFIGS_PATH = EXPERIMENTS_PATH / "kge_configs"
KGE_MODELS_PATH = EXPERIMENTS_PATH / "kge_models"


def read_experimental_setup() -> List[Dict]:
    """Read the experimental setup from comparison_setup.csv"""
    experiments = []
    with open(COMPARISON_SETUP, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            experiments.append(row)
    return experiments


def check_file_exists(path: Path) -> bool:
    """Check if a file exists and is non-empty"""
    return path.exists() and path.stat().st_size > 0


def get_experiment_status(kg_name: str, kge_model: str, lpx_config: str) -> Dict[str, bool]:
    """Check the status of all steps for a given experiment"""
    lpx = json.loads(lpx_config)
    method = lpx['method']
    summarization = lpx['summarization']
    
    status = {}
    
    # Check KGE config
    config_file = KGE_CONFIGS_PATH / f"{kg_name}_{kge_model}.json"
    status['config'] = check_file_exists(config_file)
    
    # Check KGE model
    model_file = KGE_MODELS_PATH / f"{kg_name}_{kge_model}.pt"
    status['model'] = check_file_exists(model_file)
    
    # Check predictions (rankings)
    predictions_file = PREDICTIONS_PATH / f"{kg_name}_{kge_model}.csv"
    status['predictions'] = check_file_exists(predictions_file)
    
    # Check selected predictions
    selected_file = SELECTED_PREDICTIONS_PATH / f"{kg_name}_{kge_model}.csv"
    status['selected'] = check_file_exists(selected_file)
    
    # Check explanations
    explanation_file = EXPLANATIONS_PATH / f"{kg_name}_{kge_model}_{method}_{summarization}.json"
    status['explanations'] = check_file_exists(explanation_file)
    
    return status


def format_status_symbol(status: bool) -> str:
    """Return a colored symbol for the status"""
    return "✓" if status else "✗"


def print_summary(experiments: List[Dict], verbose: bool = False, kg_filter: Optional[str] = None, 
                  model_filter: Optional[str] = None, show_complete_only: bool = False,
                  show_incomplete_only: bool = False):
    """Print a summary of experiment completion status
    
    Args:
        experiments: List of experiments from comparison_setup.csv
        verbose: If True, show detailed file paths
        kg_filter: If provided, only show experiments for this KG
        model_filter: If provided, only show experiments for this model
        show_complete_only: If True, only show completed experiments
        show_incomplete_only: If True, only show incomplete experiments
    """
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETION STATUS")
    print("="*100)
    
    # Group by KG and model for better readability
    grouped = defaultdict(list)
    for exp in experiments:
        key = (exp['kg_name'], exp['kge_model_name'])
        grouped[key].append(exp)
    
    total_experiments = len(experiments)
    completed_experiments = 0
    
    stats = {
        'config': 0,
        'model': 0,
        'predictions': 0,
        'selected': 0,
        'explanations': 0
    }
    
    incomplete_details = []
    
    for (kg_name, kge_model), exp_list in sorted(grouped.items()):
        print(f"\n{kg_name} - {kge_model}")
        print("-" * 100)
        
        # Check shared steps (config, model, predictions, selected)
        lpx_config = exp_list[0]['lpx_config']
        status = get_experiment_status(kg_name, kge_model, lpx_config)
        
        shared_complete = status['config'] and status['model'] and status['predictions'] and status['selected']
        
        print(f"  Config:      {format_status_symbol(status['config'])}")
        print(f"  Model:       {format_status_symbol(status['model'])}")
        print(f"  Predictions: {format_status_symbol(status['predictions'])}")
        print(f"  Selected:    {format_status_symbol(status['selected'])}")
        
        # Update statistics for shared steps
        for key in ['config', 'model', 'predictions', 'selected']:
            if status[key]:
                stats[key] += len(exp_list)
        
        print(f"\n  Explanations:")
        for exp in exp_list:
            lpx = json.loads(exp['lpx_config'])
            method = lpx['method']
            summarization = lpx['summarization']
            
            status = get_experiment_status(kg_name, kge_model, exp['lpx_config'])
            symbol = format_status_symbol(status['explanations'])
            
            exp_name = f"{method} (sum: {summarization})"
            print(f"    {symbol} {exp_name}")
            
            # Check if entire experiment is complete
            is_complete = all(status.values())
            if is_complete:
                completed_experiments += 1
            else:
                incomplete_steps = [k for k, v in status.items() if not v]
                incomplete_details.append({
                    'experiment': f"{kg_name}_{kge_model}_{method}_{summarization}",
                    'missing': incomplete_steps
                })
            
            if status['explanations']:
                stats['explanations'] += 1
    
    # Print overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"Total experiments:      {total_experiments}")
    print(f"Completed experiments:  {completed_experiments} ({completed_experiments/total_experiments*100:.1f}%)")
    print(f"Incomplete experiments: {total_experiments - completed_experiments}")
    print(f"\nStep completion:")
    print(f"  Configs:      {stats['config']}/{total_experiments} ({stats['config']/total_experiments*100:.1f}%)")
    print(f"  Models:       {stats['model']}/{total_experiments} ({stats['model']/total_experiments*100:.1f}%)")
    print(f"  Predictions:  {stats['predictions']}/{total_experiments} ({stats['predictions']/total_experiments*100:.1f}%)")
    print(f"  Selected:     {stats['selected']}/{total_experiments} ({stats['selected']/total_experiments*100:.1f}%)")
    print(f"  Explanations: {stats['explanations']}/{total_experiments} ({stats['explanations']/total_experiments*100:.1f}%)")
    
    # Print incomplete experiments details
    if incomplete_details:
        print("\n" + "="*100)
        print("INCOMPLETE EXPERIMENTS DETAILS")
        print("="*100)
        for item in incomplete_details:
            print(f"{item['experiment']}")
            print(f"  Missing: {', '.join(item['missing'])}")


def main():
    experiments = read_experimental_setup()
    print_summary(experiments)


if __name__ == "__main__":
    main()
