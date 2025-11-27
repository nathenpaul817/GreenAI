import os
import sys
import io
import logging
from codecarbon import OfflineEmissionsTracker
from eco2ai import Tracker as EcoTracker

# Suppress codecarbon's verbose logging
logging.getLogger("codecarbon").setLevel(logging.CRITICAL)

class UnifiedTracker:
    def __init__(self, experiment_name, tool="codecarbon", output_dir="./logs"):
        self.tool = tool
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if self.tool == "codecarbon":
            # Disable API endpoint to avoid None rounding error in codecarbon
            os.environ['CODECARBON_API_ENDPOINT'] = ''
            
            # Temporarily suppress stderr to avoid printing codecarbon's API errors
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                # Fixed ISO code ensures consistent Carbon Intensity (CI) across runs
                self.tracker = OfflineEmissionsTracker(
                    country_iso_code="USA",
                    output_dir=output_dir
                )
            finally:
                sys.stderr = old_stderr
        elif self.tool == "eco2ai":
            # Eco2AI logs locally; good for function-level granularity
            self.tracker = EcoTracker(
                project_name=experiment_name,
                file_name=f"{output_dir}/eco2ai_log.csv"
            )

    def start(self):
        self.tracker.start()

    def stop(self):
        self.tracker.stop()