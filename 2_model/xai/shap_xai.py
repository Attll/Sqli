import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

import sys
import os

# Get the directory of the current script (xai/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level (to 2_model/)
parent_dir = os.path.dirname(current_dir)

# Define the paths to the 'train' and 'preprocessing' directories
train_dir = os.path.join(parent_dir, 'train')
preprocess_dir = os.path.join(parent_dir, 'preprocessing')

# Add both directories to the Python path
if train_dir not in sys.path:
    sys.path.append(train_dir)
if preprocess_dir not in sys.path:
    sys.path.append(preprocess_dir)
class SHAPExplainer:
    """SHAP explainer for SQL injection detection"""
    
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None
        
    def initialize_explainer(self, X_train, max_samples=100):
        """Initialize SHAP explainer"""
        # Use a sample for efficiency
        if hasattr(X_train, 'toarray'):
            X_train_sample = X_train[:max_samples].toarray()
        else:
            X_train_sample = X_train[:max_samples]
        
        # Use TreeExplainer for tree-based models, else KernelExplainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_train_sample
            )
    
    def explain_instance(self, query):
        """Explain single instance"""
        # Preprocess
        query_cleaned = self.preprocessor.clean_query(query)
        X = self.preprocessor.transform_queries([query_cleaned])
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Get SHAP values
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values_instance = shap_values[1][0]  # SQLi class
        else:
            shap_values_instance = shap_values[0]
        
        return {
            'query': query,
            'prediction': int(prediction),
            'prediction_label': 'SQLi' if prediction == 1 else 'Normal',
            'probability': {
                'normal': float(probability[0]),
                'sqli': float(probability[1])
            },
            'shap_values': shap_values_instance,
            'feature_names': self.feature_names,
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
        }
    
    def visualize_waterfall(self, explanation_result, save_path=None):
        """Visualize waterfall plot"""
        import shap
        import matplotlib.pyplot as plt # Make sure this is imported
        
        # 1. Get the multi-class values from the result dictionary
        shap_values_multi = explanation_result['shap_values']
        base_value_multi = explanation_result['base_value']
        
        # 2. Get the prediction (0 or 1) to know which class to plot
        prediction = explanation_result['prediction']

        # 3. Select the values for ONLY the predicted class
        shap_values_single = shap_values_multi[:, prediction]
        base_value_single = base_value_multi[prediction]

        # 4. Create a NEW, SINGLE-CLASS Explanation object
        explanation_single = shap.Explanation(
            values=shap_values_single,
            base_values=base_value_single,
            data=shap_values_single,  # Use the single-class values for data
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        
        # 5. Plot the new, single-class explanation
        shap.plots.waterfall(explanation_single, show=False)
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_force_plot(self, explanation_result, save_path=None):
        """Visualize force plot"""
        shap_values = explanation_result['shap_values']
        base_value = explanation_result['base_value']
        
        # Get top features
        top_indices = np.argsort(np.abs(shap_values))[-20:]
        
        force_plot = shap.force_plot(
            base_value,
            shap_values[top_indices],
            feature_names=[self.feature_names[i] for i in top_indices],
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def global_feature_importance(self, X_test, max_samples=100):
        """Global feature importance using SHAP"""
        if hasattr(X_test, 'toarray'):
            X_test_sample = X_test[:max_samples].toarray()
        else:
            X_test_sample = X_test[:max_samples]
        
        shap_values = self.explainer.shap_values(X_test_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # SQLi class
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig('2_model/xai/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test_sample,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig('2_model/xai/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()


def demo_shap():
    """Demo SHAP explanations"""
    
    # Load model and preprocessor
    from train import SQLiDetector
    from preprocessing import SQLiPreprocessor
    
    # Load best model
    with open('2_model/models/best_model.txt', 'r') as f:
        best_model_type = f.read().strip()
    
    detector = SQLiDetector(model_type=best_model_type)
    detector.load('2_model/models')
    
    preprocessor = SQLiPreprocessor()
    preprocessor.load('1_data/processed')
    
    # Load feature names
    import json
    with open('1_data/processed/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    feature_names = feature_info['text_features'] + feature_info['manual_features']
    
    # Load data
    X_train = joblib.load('1_data/processed/X_train.pkl')
    X_test = joblib.load('1_data/processed/X_test.pkl')

    # Initialize explainer
    print("Initializing SHAP explainer...")
    shap_explainer = SHAPExplainer(detector.model, preprocessor, feature_names)
    shap_explainer.initialize_explainer(X_train)
    
    # Generate global feature importance
    print("Generating global feature importance...")
    shap_explainer.global_feature_importance(X_test)
    
    # Test queries
    test_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE id = 1' OR '1'='1",
        "'; DROP TABLE users; --"
    ]
    
    Path('2_model/xai/shap_explanations').mkdir(parents=True, exist_ok=True)
    
    for i, query in enumerate(test_queries):
        print(f"\nExplaining query {i+1}: {query}")
        
        result = shap_explainer.explain_instance(query)
        
        print(f"Prediction: {result['prediction_label']}")
        print(f"Probability: {result['probability']['sqli']:.3f}")
        
        # Save visualizations
        shap_explainer.visualize_waterfall(
            result,
            save_path=f'2_model/xai/shap_explanations/waterfall_{i+1}.png'
        )
    
    print("\nSHAP explanations saved!")


if __name__ == "__main__":
    demo_shap()