import lime
import lime.lime_tabular
import numpy as np
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
class LIMEExplainer:
    """LIME explainer for SQL injection detection"""
    
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None
        
    def initialize_explainer(self, X_train):
        """Initialize LIME explainer"""
        # Convert sparse to dense
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=['Normal', 'SQLi'],
            discretize_continuous=False,
            mode='classification'
        )
    
    def explain_instance(self, query, num_features=10):
        """Explain a single query"""
        # Preprocess query
        query_cleaned = self.preprocessor.clean_query(query)
        X = self.preprocessor.transform_queries([query_cleaned])
        
        # Convert to dense
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            X[0],
            self.model.predict_proba,
            num_features=num_features,
            top_labels=2
        )
        
        return {
            'query': query,
            'prediction': int(prediction),
            'prediction_label': 'SQLi' if prediction == 1 else 'Normal',
            'probability': {
                'normal': float(probability[0]),
                'sqli': float(probability[1])
            },
            'explanation': explanation.as_list(label=prediction),
            'explanation_object': explanation
        }
    
    def visualize_explanation(self, explanation_result, save_path=None):
        """Visualize LIME explanation"""
        exp = explanation_result['explanation_object']
        
        # Show in notebook or save
        if save_path:
            exp.save_to_file(save_path)
        else:
            exp.show_in_notebook(show_table=True)


def demo_lime():
    """Demo LIME explanations"""
    
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
    
    # Load training data for LIME
    X_train = joblib.load('1_data/processed/X_train.pkl')
    
    # Initialize explainer
    lime_explainer = LIMEExplainer(detector.model, preprocessor, feature_names)
    lime_explainer.initialize_explainer(X_train)
    
    # Test queries
    test_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE id = 1' OR '1'='1",
        "SELECT * FROM users WHERE username = 'admin' AND password = 'pass123'",
        "SELECT * FROM users WHERE id = 1 UNION SELECT null, username, password FROM admin",
        "'; DROP TABLE users; --"
    ]

    Path('2_model/xai/lime_explanations').mkdir(parents=True, exist_ok=True)

    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {query}")
        print(f"{'='*60}")
        
        result = lime_explainer.explain_instance(query)
        
        print(f"Prediction: {result['prediction_label']}")
        print(f"Probability: Normal={result['probability']['normal']:.3f}, "
              f"SQLi={result['probability']['sqli']:.3f}")
        print("\nTop contributing features:")
        for feature, weight in result['explanation'][:10]:
            print(f"  {feature}: {weight:.4f}")
        
        # Save visualization
        lime_explainer.visualize_explanation(
            result,
            save_path=f'2_model/xai/lime_explanations/query_{i+1}.html'
        )
    
    print("\nLIME explanations saved!")


if __name__ == "__main__":
    demo_lime()