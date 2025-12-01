#!/usr/bin/env python3
import subprocess
import sys
import itertools
import csv
import os
import json
from datetime import datetime

def run_experiments():
    """
    Run all combinations of parameters for greenAI.py
    Models: logreg, forest
    Tools: codecarbon, eco2ai
    Batch sizes: 4, 16, 64, 256
    Precision: fp32, fp16
    """
    
    models = ["logreg", "forest"]
    tools = ["codecarbon", "eco2ai"]
    batch_sizes = [4, 16, 64, 256]
    precisions = ["fp32", "fp16"]
    
    # Generate all combinations
    combinations = list(itertools.product(models, tools, batch_sizes, precisions))
    
    total_runs = len(combinations)
    print(f"Total experiment runs: {total_runs}")
    print("=" * 80)
    
    # Create experiment_results directory if it doesn't exist
    os.makedirs("experiment_results", exist_ok=True)
    
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"experiment_results/results_{timestamp}.csv"
    
    # Initialize counters
    completed = 0
    failed = 0
    results = []
    
    for i, (model, tool, batch_size, precision) in enumerate(combinations, 1):
        print(f"\n[{i}/{total_runs}] Running: model={model}, tool={tool}, batch_size={batch_size}, precision={precision}")
        print("-" * 80)
        
        cmd = [
            "python3",
            "greenAI.py",
            "--model", model,
            "--tool", tool,
            "--batch-size", str(batch_size),
            "--precision", precision
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            completed += 1
            
            # Parse output for metrics
            output = result.stdout + result.stderr
            
            # Extract metrics based on tool
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "tool": tool,
                "batch_size": batch_size,
                "precision": precision,
                "runtime_seconds": "N/A"
            }
            
            # Look for JSON results in output
            if "EXPERIMENT_RESULTS:" in output:
                try:
                    json_str = output.split("EXPERIMENT_RESULTS:")[1].strip().split('\n')[0]
                    experiment_data = json.loads(json_str)
                    metrics.update(experiment_data)
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"Warning: Could not parse results JSON: {e}")
            
            results.append(metrics)
            print(f"✓ Completed [{i}/{total_runs}]")
            
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"✗ Failed [{i}/{total_runs}]: {e}")
            
            # Still record the failed run
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "tool": tool,
                "batch_size": batch_size,
                "precision": precision,
                "runtime_seconds": "FAILED"
            }
            results.append(metrics)
    
    # Write results to CSV
    try:
        if results:
            # Gather all unique fieldnames from results
            fieldnames = set()
            fieldnames.update(["timestamp", "model", "tool", "batch_size", "precision", "runtime_seconds"])
            
            for result in results:
                fieldnames.update(result.keys())
            
            fieldnames = sorted(list(fieldnames))
            
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval="N/A")
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResults saved to: {csv_filename}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")
    
    print("\n" + "=" * 80)
    print(f"Experiments Summary:")
    print(f"  Total runs: {total_runs}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = run_experiments()
    sys.exit(0 if success else 1)
