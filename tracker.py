import os
import sys
import io
import csv
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
    
    def get_results(self):
        """Get results based on the tracking tool used"""
        results = {}
        
        if self.tool == "codecarbon":
            # CodeCarbon returns results directly from tracker.stop()
            try:
                emissions = self.tracker.stop()
                if emissions:
                    results = {
                        "emissions": emissions,
                        "emissions_rate": getattr(self.tracker, 'emissions_rate', 'N/A'),
                        "energy_consumed": getattr(self.tracker, 'energy_consumed', 'N/A')
                    }
            except Exception as e:
                print(f"Error getting codecarbon results: {e}")
                
        elif self.tool == "eco2ai":
            # Eco2AI logs to file; need to read the latest entry
            try:
                import csv
                eco2ai_file = f"{self.output_dir}/eco2ai_log.csv"
                if os.path.exists(eco2ai_file):
                    with open(eco2ai_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Get the last row
                            results = {
                                "power_consumption_kwh": row.get('power_consumption(kWh)', 'N/A'),
                                "co2_emissions_kg": row.get('CO2_emissions(kg)', 'N/A')
                            }
            except Exception as e:
                print(f"Error getting eco2ai results: {e}")
        
        return results