
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

# Use the UnifiedTracker defined above
from tracker import UnifiedTracker 

def run_classical_experiment(model_type, tool):
   
    print("Loading UCI Adult dataset...")
    X, y = fetch_openml(name="adult", version=2, return_X_y=True, as_frame=True)
    
    #Preprocess: Handle missing values and encode categorical features
    print("Preprocessing data...")
    
    # Convert categorical columns to strings first
    for col in X.columns:
        if X[col].dtype == 'category':
            X[col] = X[col].astype(str)
    
    # Fill missing values and encode
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Ensure all values are numeric
    X = X.astype(float).values
    
    # Encode target variable if it's categorical
    if y.dtype == 'category':
        y = y.astype(str)
    y = y.fillna('Unknown').astype(str)
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # Initialize Models
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "forest":
        model = RandomForestClassifier(n_estimators=100)

    # Energy Tracking 
    tracker = UnifiedTracker(experiment_name=f"Tier1_{model_type}", tool=tool)
    
    print(f"Starting training for {model_type}...")
    tracker.start()
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    tracker.stop()
    
    print(f"Training completed in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "forest"], required=True)
    parser.add_argument("--tool", choices=["codecarbon", "eco2ai"], default="codecarbon")
    args = parser.parse_args()
    
    run_classical_experiment(args.model, args.tool)