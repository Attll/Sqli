# --- ADD THESE 3 LINES ---
import sys
import os
import json

# --- ADD THIS BLOCK TO FIX THE IMPORT ---
# Get the directory of the current script (evaluate/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level (to 2_model/)
parent_dir = os.path.dirname(current_dir)
# Define the path to the 'train' directory
train_dir = os.path.join(parent_dir, 'train')

# Add the 'train' directory to the Python path
if train_dir not in sys.path:
    sys.path.append(train_dir)
# --- END OF FIX ---

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

# This line will now work
from train import SQLiDetector


def evaluate_detailed(model_path='2_model/models', data_path='1_data/processed'):
    """Detailed model evaluation"""
    
    # Load best model
    with open(f"{model_path}/best_model.txt", 'r') as f:
        best_model_type = f.read().strip()
    
    print(f"Loading {best_model_type} model...")
    detector = SQLiDetector(model_type=best_model_type)
    detector.load(model_path)
    
    # Load test data
    X_test = joblib.load(f"{data_path}/X_test.pkl")
    y_test = joblib.load(f"{data_path}/y_test.pkl")
    X_test_raw = joblib.load(f"{data_path}/X_test_raw.pkl")
    
    # Predictions
    y_pred = detector.predict(X_test)
    y_pred_proba = detector.predict_proba(X_test)[:, 1]

    # --- FIX 2: LOAD SAVED METRICS ---
    # The detector.metrics object is empty on a fresh load.
    # We must load the metrics saved by train.py.
    with open(f"{model_path}/metrics_{best_model_type}.json", 'r') as f:
        metrics = json.load(f)
    roc_auc = metrics.get('roc_auc', 0)  # Get the saved ROC-AUC score
    # --- END OF FIX 2 ---
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'AP = {ap_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_path}/precision_recall_curve.png', dpi=300)
    plt.close()
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    # --- Use the loaded roc_auc value ---
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})') 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_path}/roc_curve.png', dpi=300)
    plt.close()
    
    # Error Analysis - False Positives and False Negatives
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    
    print(f"\nFalse Positives: {len(fp_indices)}")
    print("Sample FP queries:")
    for idx in fp_indices[:5]:
        print(f"  - {X_test_raw[idx]} (prob: {y_pred_proba[idx]:.3f})")
    
    print(f"\nFalse Negatives: {len(fn_indices)}")
    print("Sample FN queries:")
    for idx in fn_indices[:5]:
        print(f"  - {X_test_raw[idx]} (prob: {y_pred_proba[idx]:.3f})")
    
    # Save error analysis
    error_analysis = {
        'false_positives': [
            {'query': X_test_raw[idx], 'probability': float(y_pred_proba[idx])}
            for idx in fp_indices
        ],
        'false_negatives': [
            {'query': X_test_raw[idx], 'probability': float(y_pred_proba[idx])}
            for idx in fn_indices
        ]
    }
    
    # Note: 'import json' was moved to the top
    with open(f'{model_path}/error_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate_detailed()