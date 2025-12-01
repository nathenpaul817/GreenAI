#!/usr/bin/env python3
"""
Run all combinations of parameters for medium-scale ResNet18 training experiments
and save results to CSV file under experiment_results folder.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Define parameter combinations
TOOLS = ["codecarbon", "eco2ai"]
BATCH_SIZES = [4, 16, 64, 256]
PRECISIONS = ["fp32", "fp16"]

# Result columns for each tool
CODECARBON_COLUMNS = [
    "timestamp", "tool", "batch_size", "precision", 
    "emissions", "emissions_rate", "energy_consumed",
    "runtime_seconds", "status"
]

ECO2AI_COLUMNS = [
    "timestamp", "tool", "batch_size", "precision",
    "power_consumption_kwh", "co2_emissions_kg",
    "runtime_seconds", "status"
]


def run_experiment(tool, batch_size, precision, image_dir="./ILSVRC_train", epochs=5):
    """
    Run a single experiment with specified parameters.
    
    Args:
        tool: "codecarbon" or "eco2ai"
        batch_size: batch size for training
        precision: "fp32" or "fp16"
        image_dir: path to training images
        epochs: number of training epochs
    
    Returns:
        dict: Results from the training run, or None if failed
    """
    cmd = [
        "python3",
        "greenAI_medium.py",
        "--tool", tool,
        "--batch-size", str(batch_size),
        "--precision", precision,
        "--epochs", str(epochs),
        "--image-dir", image_dir
    ]
    
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    try:
        # Run the training script and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Parse the EXPERIMENT_RESULTS from output
        for line in result.stdout.split('\n'):
            if "EXPERIMENT_RESULTS:" in line:
                try:
                    json_str = line.split("EXPERIMENT_RESULTS:", 1)[1].strip()
                    results = json.loads(json_str)
                    results['status'] = 'success'
                    return results
                except json.JSONDecodeError as e:
                    print(f"Error parsing results JSON: {e}")
                    print(f"Line: {line}")
                    return None
        
        print(f"Warning: No EXPERIMENT_RESULTS found in output")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Error: Experiment timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Error running experiment: {e}")
        return None


def extract_codecarbon_metrics(results):
    """Extract CodeCarbon metrics from results dictionary."""
    return {
        "emissions": results.get("emissions", "N/A"),
        "emissions_rate": results.get("emissions_rate", "N/A"),
        "energy_consumed": results.get("energy_consumed", "N/A"),
    }


def extract_eco2ai_metrics(results):
    """Extract Eco2AI metrics from results dictionary."""
    return {
        "power_consumption_kwh": results.get("power_consumption_kwh", "N/A"),
        "co2_emissions_kg": results.get("co2_emissions_kg", "N/A"),
    }


def save_results_codecarbon(csv_file, tool, batch_size, precision, results, runtime):
    """Save CodeCarbon experiment results to CSV."""
    metrics = extract_codecarbon_metrics(results)
    
    row = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool,
        "batch_size": batch_size,
        "precision": precision,
        "emissions": metrics["emissions"],
        "emissions_rate": metrics["emissions_rate"],
        "energy_consumed": metrics["energy_consumed"],
        "runtime_seconds": runtime,
        "status": results.get("status", "unknown")
    }
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CODECARBON_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"Results saved: {row}")


def save_results_eco2ai(csv_file, tool, batch_size, precision, results, runtime):
    """Save Eco2AI experiment results to CSV."""
    metrics = extract_eco2ai_metrics(results)
    
    row = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool,
        "batch_size": batch_size,
        "precision": precision,
        "power_consumption_kwh": metrics["power_consumption_kwh"],
        "co2_emissions_kg": metrics["co2_emissions_kg"],
        "runtime_seconds": runtime,
        "status": results.get("status", "unknown")
    }
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ECO2AI_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"Results saved: {row}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all combinations of experiments for ResNet18 training"
    )
    parser.add_argument(
        "--image-dir",
        default="./ILSVRC_train",
        help="Path to directory containing training images"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs per experiment"
    )
    parser.add_argument(
        "--output-dir",
        default="./experiment_results",
        help="Directory to save results CSV"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed experiment"
    )
    
    args = parser.parse_args()
    
    # Verify image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' not found!")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create separate CSV files for each tool
    codecarbon_csv = output_dir / f"results_medium_codecarbon_{timestamp}.csv"
    eco2ai_csv = output_dir / f"results_medium_eco2ai_{timestamp}.csv"
    
    # Track experiment progress
    total_experiments = len(TOOLS) * len(BATCH_SIZES) * len(PRECISIONS)
    completed = 0
    failed = 0
    
    print(f"\n{'='*70}")
    print(f"ResNet18 Medium-Scale Experiments Runner")
    print(f"{'='*70}")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Tools: {TOOLS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Precisions: {PRECISIONS}")
    print(f"Output directory: {output_dir}")
    print(f"Codecarbon results: {codecarbon_csv.name}")
    print(f"Eco2AI results: {eco2ai_csv.name}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    try:
        # Run all combinations
        for tool in TOOLS:
            for batch_size in BATCH_SIZES:
                for precision in PRECISIONS:
                    try:
                        # Run experiment
                        runtime = None
                        results = run_experiment(
                            tool=tool,
                            batch_size=batch_size,
                            precision=precision,
                            image_dir=args.image_dir,
                            epochs=args.epochs
                        )
                        
                        if results and 'runtime_seconds' in results:
                            runtime = results['runtime_seconds']
                        
                        if results:
                            # Save results to appropriate CSV
                            if tool == "codecarbon":
                                save_results_codecarbon(
                                    codecarbon_csv, tool, batch_size, precision, results, runtime
                                )
                            elif tool == "eco2ai":
                                save_results_eco2ai(
                                    eco2ai_csv, tool, batch_size, precision, results, runtime
                                )
                            completed += 1
                        else:
                            print(f"Failed to get results for tool={tool}, "
                                  f"batch_size={batch_size}, precision={precision}")
                            failed += 1
                    
                    except Exception as e:
                        print(f"Error in experiment tool={tool}, batch_size={batch_size}, "
                              f"precision={precision}: {e}")
                        failed += 1
                    
                    # Print progress
                    progress = completed + failed
                    print(f"\nProgress: {progress}/{total_experiments} experiments completed")
                    print(f"Completed: {completed}, Failed: {failed}\n")
    
    except KeyboardInterrupt:
        print("\n\nExperiment run interrupted by user")
    
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"Experiment Run Summary")
        print(f"{'='*70}")
        print(f"Total experiments: {total_experiments}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration}")
        print(f"Codecarbon results saved to: {codecarbon_csv}")
        print(f"Eco2AI results saved to: {eco2ai_csv}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
