import joblib
import json
import time
from pathlib import Path
import sys
import os # <-- 1. Import os for path manipulation

# --- START FIX ---

# 2. Define paths relative to THIS file's location
# __file__ is 'detector.py'. os.path.dirname(__file__) is '3_middleware'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is one level up (the 'SQLIDEV' folder)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# 3. Add the folder containing preprocessing.py to the system path
# The correct path is '../2_model/preprocessing'
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, '2_model', 'preprocessing')
sys.path.append(PREPROCESSING_DIR)

# 4. The import can now find 'preprocessing.py' in that directory
from preprocessing import SQLiPreprocessor

# --- END FIX ---


class SQLiDetectorMiddleware:
    """Middleware for detecting SQL injection attacks"""

    # --- START FIX ---
    # 5. Update default paths to use underscores and the absolute PROJECT_ROOT
    def __init__(self, 
                 model_path=os.path.join(PROJECT_ROOT, '2_model', 'models'), 
                 data_path=os.path.join(PROJECT_ROOT, '1_data', 'processed')):
    # --- END FIX ---
        """Initialize detector"""
        self.load_model(model_path, data_path)
        self.request_history = []

    def load_model(self, model_path, data_path):
        """Load trained model and preprocessor"""
        # Load best model type
        with open(f'{model_path}/best_model.txt', 'r') as f:
            model_type = f.read().strip()

        # Load model
        self.model = joblib.load(f'{model_path}/model_{model_type}.pkl')

        # Load preprocessor
        self.preprocessor = SQLiPreprocessor()
        self.preprocessor.load(data_path)

        # Load metrics
        with open(f'{model_path}/metrics_{model_type}.json', 'r') as f:
            self.metrics = json.load(f)

        print(f"Loaded {model_type} model")
        print(f"Model accuracy: {self.metrics['accuracy']:.4f}")

    def detect(self, query, threshold=0.5):
        """Detect if query is malicious"""
        start_time = time.time()

        # Preprocess
        query_cleaned = self.preprocessor.clean_query(query)
        X = self.preprocessor.transform_queries([query_cleaned])

        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]

        detection_time = time.time() - start_time

        # Build result
        result = {
            'query': query,
            'is_malicious': bool(prediction == 1 or probability[1] > threshold),
            'confidence': float(probability[1]),
            'prediction': int(prediction),
            'threshold': threshold,
            'detection_time_ms': detection_time * 1000,
            'timestamp': time.time()
        }

        # Store in history
        self.request_history.append(result)

        return result

    def batch_detect(self, queries, threshold=0.5):
        """Detect multiple queries"""
        results = []
        for query in queries:
            result = self.detect(query, threshold)
            results.append(result)
        return results

    def get_statistics(self):
        """Get detection statistics"""
        if not self.request_history:
            return {}

        total_requests = len(self.request_history)
        malicious_count = sum(1 for r in self.request_history if r['is_malicious'])

        avg_detection_time = sum(r['detection_time_ms'] for r in self.request_history) / total_requests
        avg_confidence = sum(r['confidence'] for r in self.request_history) / total_requests

        return {
            'total_requests': total_requests,
            'malicious_detected': malicious_count,
            'benign_requests': total_requests - malicious_count,
            'detection_rate': malicious_count / total_requests if total_requests > 0 else 0,
            'avg_detection_time_ms': avg_detection_time,
            'avg_confidence': avg_confidence
        }

    def clear_history(self):
        """Clear request history"""
        self.request_history = []

# Test
if __name__ == "__main__":
    # This will now work even if you run: python 3_middleware/detector.py
    detector = SQLiDetectorMiddleware() 

    test_queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE id = 1' OR '1'='1",
        "SELECT * FROM products WHERE name = 'laptop'",
        "'; DROP TABLE users; --",
        "SELECT * FROM users WHERE username = 'admin' AND password = 'pass123'"
    ]

    print("\nTesting SQL Injection Detector\n" + "="*60)
    for query in test_queries:
        result = detector.detect(query)
        status = "🚨 BLOCKED" if result['is_malicious'] else "✅ ALLOWED"
        print(f"\n{status}")
        print(f"Query: {query}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Detection time: {result['detection_time_ms']:.2f}ms")

    print("\n" + "="*60)
    stats = detector.get_statistics()
    print(f"\nStatistics:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Malicious detected: {stats['malicious_detected']}")
    print(f"Detection rate: {stats['detection_rate']:.2%}")
    print(f"Avg detection time: {stats['avg_detection_time_ms']:.2f}ms")