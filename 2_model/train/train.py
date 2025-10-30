import numpy as np
import joblib
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.sparse import vstack

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


class SQLiDetector:
    """SQL Injection Detection Model"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._initialize_model(model_type)
        self.metrics = {}
        
    def _initialize_model(self, model_type):
        """Initialize ML model"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'naive_bayes': MultinomialNB(alpha=0.1)
        }
        
        if model_type not in models:
            raise ValueError(f"Model type must be one of {list(models.keys())}")
        
        return models[model_type]
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        print(f"Training data shape: {X_train.shape}")
        
        self.model.fit(X_train, y_train)
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def evaluate(self, X_test, y_test, output_path='.'):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print("\nTest Metrics:")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'SQLi']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'SQLi'],
                   yticklabels=['Normal', 'SQLi'])
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # --- PATH FIXED ---
        # Use the provided output_path argument
        plt.savefig(f'{output_path}/confusion_matrix_{self.model_type}.png', dpi=300)
        plt.close()
        
        return self.metrics
    
    def predict(self, X):
        """Predict on new data"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names, top_n=20, output_path='.'):
        """Get feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top {top_n} Feature Importances - {self.model_type}')
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # --- PATH FIXED ---
            # Use the provided output_path argument
            plt.savefig(f'{output_path}/feature_importance_{self.model_type}.png', dpi=300)
            plt.close()
            
            return dict(zip([feature_names[i] for i in indices], 
                          importances[indices].tolist()))
        return None
    
    def save(self, path):
        """Save model"""
        joblib.dump(self.model, f"{path}/model_{self.model_type}.pkl")
        
        # Save metrics
        self.metrics['model_type'] = self.model_type
        self.metrics['timestamp'] = datetime.now().isoformat()
        
        with open(f"{path}/metrics_{self.model_type}.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, path):
        """Load model"""
        # Note: You'll need to pass the model_type to the constructor
        # before calling load, as the path depends on it.
        self.model = joblib.load(f"{path}/model_{self.model_type}.pkl")


# --- PATHS FIXED ---
# Default paths are now relative to the project root (e.g., C:\Projects\SqliDev)
def train_all_models(data_path='1_data/processed', output_path='2_model/models'):
    """Train and compare multiple models"""
    
    print("Loading processed data...")
    X_train = joblib.load(f"{data_path}/X_train.pkl")
    X_val = joblib.load(f"{data_path}/X_val.pkl")
    X_test = joblib.load(f"{data_path}/X_test.pkl")
    y_train = joblib.load(f"{data_path}/y_train.pkl")
    y_val = joblib.load(f"{data_path}/y_val.pkl")
    y_test = joblib.load(f"{data_path}/y_test.pkl")
    
    # Load feature names
    with open(f"{data_path}/feature_info.json", 'r') as f:
        feature_info = json.load(f)
    feature_names = feature_info['text_features'] + feature_info['manual_features']
    
    # Combine train and val for final training
    X_train_full = vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    print(f"\nFull training set: {X_train_full.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train multiple models
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    results = {}
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        detector = SQLiDetector(model_type=model_type)
        detector.train(X_train_full, y_train_full)
        
        # --- PATHS FIXED ---
        # Pass the output_path to the evaluate method
        metrics = detector.evaluate(X_test, y_test, output_path=output_path)
        
        # Get feature importance
        if model_type in ['random_forest', 'gradient_boosting']:
            # --- PATHS FIXED ---
            # Pass the output_path to the get_feature_importance method
            importance = detector.get_feature_importance(feature_names, output_path=output_path)
        
        detector.save(output_path)
        results[model_type] = metrics
    
    # Compare models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    comparison_df = []
    for model_type, metrics in results.items():
        comparison_df.append({
            'Model': model_type,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_df)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(f"{output_path}/model_comparison.csv", index=False)
    
    # Select best model based on F1-score
    best_model_type = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    print(f"\nBest model: {best_model_type}")
    
    # Save best model info
    with open(f"{output_path}/best_model.txt", 'w') as f:
        f.write(best_model_type)
    
    return results, best_model_type


if __name__ == "__main__":
    # This will now use the corrected default paths
    results, best_model = train_all_models()