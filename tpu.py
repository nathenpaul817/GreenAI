
"""
Script to run run_experiments_medium.py on a TPU in Google Colab.
This script is intended to be run in a Colab environment with TPU runtime enabled.
It will set up the environment, install dependencies, and execute the experiment runner.
"""

import os
import sys
import subprocess

def is_colab():
	try:
		import google.colab  # type: ignore
		return True
	except ImportError:
		return False

def setup_tpu():
	# Check TPU availability
	try:
		import torch_xla
		import torch_xla.core.xla_model as xm
		print("TPU is available.")
	except ImportError:
		print("torch_xla not found. Installing torch_xla for TPU support...")
		subprocess.check_call([
			sys.executable, "-m", "pip", "install", "torch_xla[tpu]", "--extra-index-url", "https://pypi.org/simple"
		])
		print("torch_xla installed.")

def install_requirements():
	req_file = "requirements.txt"
	if os.path.exists(req_file):
		print(f"Installing requirements from {req_file}...")
		subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
	else:
		print(f"No {req_file} found. Skipping requirements installation.")

def run_experiments_medium():
	script = "run_experiments_medium.py"
	if not os.path.exists(script):
		print(f"{script} not found in current directory.")
		sys.exit(1)
	print(f"Running {script} on TPU...")
	# You may want to adjust arguments as needed
	cmd = [sys.executable, script, "--image-dir", "./ILSVRC_train", "--epochs", "5"]
	subprocess.check_call(cmd)

def main():
	if not is_colab():
		print("This script is intended to be run in Google Colab with TPU runtime enabled.")
		sys.exit(1)
	print("Setting up TPU environment in Colab...")
	setup_tpu()
	install_requirements()
	run_experiments_medium()

if __name__ == "__main__":
	main()
