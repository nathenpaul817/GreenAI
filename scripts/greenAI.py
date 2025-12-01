
import argparse
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import time
import warnings

# Use the UnifiedTracker defined above
from tracker import UnifiedTracker 

def run_classical_experiment(model_type, tool, epochs, batch_size=32, precision='fp32'):
   
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

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize Models
    if model_type == "logreg":
        model = LogisticRegression(max_iter=5000, solver='lbfgs')
    elif model_type == "forest":
        model = RandomForestClassifier(n_estimators=100)

    # Energy Tracking 
    tracker = UnifiedTracker(experiment_name=f"Tier1_{model_type}", tool=tool)
    
    # Suppress FutureWarnings from eco2ai library
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print(f"Starting training for {model_type} ({epochs} epochs)...")
    tracker.start()
    start_time = time.time()
    
    for epoch in range(epochs):
        model.fit(X_train, y_train)
        if epoch % max(1, epochs // 10) == 0:
            score = model.score(X_test, y_test)
            print(f"Epoch {epoch + 1}/{epochs} | Test Score: {score:.4f}")
    
    end_time = time.time()
    tracker.stop()
    runtime = end_time - start_time
    
    print(f"Training completed in {runtime:.2f}s")
    
    # Get and print results as JSON
    results = tracker.get_results()
    results['runtime_seconds'] = runtime
    print(f"EXPERIMENT_RESULTS: {json.dumps(results)}")
    
    return tracker, runtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "forest"], required=True)
    parser.add_argument("--tool", choices=["codecarbon", "eco2ai"], default="codecarbon")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()
    
    run_classical_experiment(args.model, args.tool, args.epochs, args.batch_size, args.precision)