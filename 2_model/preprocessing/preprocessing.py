import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
import json
from pathlib import Path

class SQLiPreprocessor:
    """Preprocess SQL queries for ML model"""
    
    def __init__(self):
        self.vectorizer = None
        self.feature_names = None
        
    def clean_query(self, query):
        """Clean and normalize SQL query"""
        if pd.isna(query):
            return ""
        
        # Convert to lowercase
        query = str(query).lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Normalize common SQLi patterns
        query = query.strip()
        
        return query
    
    def extract_features(self, query):
        """Extract manual features from query"""
        features = {}
        
        # Length features
        features['length'] = len(query)
        features['word_count'] = len(query.split())
        
        # Special characters
        features['special_char_count'] = len(re.findall(r'[^a-zA-Z0-9\s]', query))
        features['quote_count'] = query.count("'") + query.count('"')
        features['dash_count'] = query.count('--')
        features['semicolon_count'] = query.count(';')
        features['comment_count'] = query.count('/*') + query.count('*/')
        
        # SQL keywords (boolean presence)
        sql_keywords = ['select', 'union', 'insert', 'update', 'delete', 
                       'drop', 'create', 'alter', 'exec', 'execute',
                       'script', 'javascript', 'onerror', 'onload']
        
        for keyword in sql_keywords:
            features[f'has_{keyword}'] = int(keyword in query)
        
        # Suspicious patterns
        features['has_or_1_1'] = int(bool(re.search(r"or\s+['\"]?1['\"]?\s*=\s*['\"]?1", query)))
        features['has_comment'] = int('--' in query or '/*' in query)
        features['has_union_select'] = int(bool(re.search(r'union.*select', query)))
        features['has_always_true'] = int(bool(re.search(r"(or|and)\s+['\"]?1['\"]?\s*=\s*['\"]?1", query)))
        
        # Character ratios
        if len(query) > 0:
            features['digit_ratio'] = sum(c.isdigit() for c in query) / len(query)
            features['alpha_ratio'] = sum(c.isalpha() for c in query) / len(query)
            features['special_ratio'] = features['special_char_count'] / len(query)
        else:
            features['digit_ratio'] = 0
            features['alpha_ratio'] = 0
            features['special_ratio'] = 0
        
        return features
    
    def fit_vectorizer(self, queries, method='tfidf', max_features=5000):
        """Fit text vectorizer"""
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),
                analyzer='char_wb',
                min_df=2,
                max_df=0.95
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),
                analyzer='char_wb',
                min_df=2,
                max_df=0.95
            )
        
        self.vectorizer.fit(queries)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
    def transform_queries(self, queries):
        """Transform queries to feature vectors"""
        # Text features
        text_features = self.vectorizer.transform(queries)
        
        # Manual features
        manual_features = []
        for query in queries:
            features = self.extract_features(query)
            manual_features.append(list(features.values()))
        
        manual_features = np.array(manual_features)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        combined_features = hstack([text_features, csr_matrix(manual_features)])
        
        return combined_features
    
    def save(self, path):
        """Save preprocessor"""
        joblib.dump(self.vectorizer, f"{path}/vectorizer.pkl")
        
        # Save feature info
        feature_info = {
            'text_features': self.feature_names.tolist() if self.feature_names is not None else [],
            'manual_features': list(self.extract_features("dummy").keys())
        }
        with open(f"{path}/feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
    
    def load(self, path):
        """Load preprocessor"""
        self.vectorizer = joblib.load(f"{path}/vectorizer.pkl")
        self.feature_names = self.vectorizer.get_feature_names_out()


def preprocess_data(input_path, output_path, test_size=0.2, val_size=0.1):
    """Main preprocessing pipeline"""
    
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Class distribution:\n{df['Label'].value_counts()}")
    
    # Clean queries
    print("\nCleaning queries...")
    preprocessor = SQLiPreprocessor()
    df['Query_cleaned'] = df['Query'].apply(preprocessor.clean_query)
    
    # Remove duplicates
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=['Query_cleaned'])
    print(f"After deduplication: {df.shape}")
    
    # Remove empty queries
    df = df[df['Query_cleaned'].str.len() > 0]
    print(f"After removing empty: {df.shape}")
    
    # Split data
    print("\nSplitting data...")
    X = df['Query_cleaned'].values
    y = df['Label'].values
    
    # Train/temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), 
        stratify=y, random_state=42
    )
    
    # Val/test split
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val_size),
        stratify=y_temp, random_state=42
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Fit vectorizer on training data
    print("\nFitting vectorizer...")
    preprocessor.fit_vectorizer(X_train)
    
    # Transform all splits
    print("Transforming data...")
    X_train_transformed = preprocessor.transform_queries(X_train)
    X_val_transformed = preprocessor.transform_queries(X_val)
    X_test_transformed = preprocessor.transform_queries(X_test)
    
    # Save processed data
    print("\nSaving processed data...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    joblib.dump(X_train_transformed, f"{output_path}/X_train.pkl")
    joblib.dump(X_val_transformed, f"{output_path}/X_val.pkl")
    joblib.dump(X_test_transformed, f"{output_path}/X_test.pkl")
    joblib.dump(y_train, f"{output_path}/y_train.pkl")
    joblib.dump(y_val, f"{output_path}/y_val.pkl")
    joblib.dump(y_test, f"{output_path}/y_test.pkl")
    
    # Save raw queries for XAI
    joblib.dump(X_train, f"{output_path}/X_train_raw.pkl")
    joblib.dump(X_val, f"{output_path}/X_val_raw.pkl")
    joblib.dump(X_test, f"{output_path}/X_test_raw.pkl")
    
    # Save preprocessor
    preprocessor.save(output_path)
    
    print("\nPreprocessing complete!")
    print(f"Feature dimensions: {X_train_transformed.shape[1]}")
    
    return preprocessor, (X_train_transformed, X_val_transformed, X_test_transformed), (y_train, y_val, y_test)


if __name__ == "__main__":
    preprocess_data(
        input_path="1_data/dataset/sqli.csv",
        output_path="1_data/processed"
    )