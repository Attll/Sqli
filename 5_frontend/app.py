from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import os
import sys

# --- [FIX 1: Path Correction] ---
# Get the absolute path to the project's root directory
# (one level up from this file's directory '5_frontend')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the '3_middleware' directory to the system path
MIDDLEWARE_PATH = os.path.join(PROJECT_ROOT, '3_middleware')
sys.path.append(MIDDLEWARE_PATH)

# Add the '2_model/preprocessing' directory for the preprocessor
PREPROCESSING_PATH = os.path.join(PROJECT_ROOT, '2_model', 'preprocessing')
sys.path.append(PREPROCESSING_PATH)

# --- [End Fix] ---

from detector import SQLiDetectorMiddleware

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
CORS(app)

# Initialize detector
try:
    detector = SQLiDetectorMiddleware()
except Exception as e:
    print(f"CRITICAL: Failed to initialize SQLiDetectorMiddleware: {e}")
    print("This might be due to missing model files or incorrect paths in 'detector.py'.")
    # You might want to exit or have the app run in a degraded state
    detector = None 

# --- [FIX 2: DB Configuration] ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'Pizza1Noodles2'), # Your password
    'database': 'sqli_demo'
}
# --- [End Fix] ---

def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Database error: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/vulnerable')
def vulnerable():
    """Vulnerable demo page"""
    return render_template('vulnerable.html')

@app.route('/protected')
def protected():
    """Protected demo page"""
    return render_template('protected.html')

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    stats = {}
    if detector:
        stats = detector.get_statistics()
    else:
        stats = {"error": "Detector not loaded"}
    return render_template('dashboard.html', stats=stats)

# ============= VULNERABLE ENDPOINTS (for demo) =============

@app.route('/api/vulnerable/login', methods=['POST'])
def vulnerable_login():
    """Vulnerable login endpoint"""
    username = request.form.get('username', '')
    password = request.form.get('password', '')
        
    # VULNERABLE SQL QUERY - DO NOT USE IN PRODUCTION
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # This is where the injection happens
            cursor.execute(query) 
            user = cursor.fetchone()
            cursor.close()
            conn.close()
                        
            return jsonify({
                'success': user is not None,
                'message': 'Login successful' if user else 'Invalid credentials',
                'user': user,
                'query': query  # Show query for demo purposes
            })
        except Exception as e:
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}',
                'query': query
            }), 500
    return jsonify({'success': False, 'message': 'Database error'}), 500

@app.route('/api/vulnerable/search', methods=['GET'])
def vulnerable_search():
    """Vulnerable product search"""
    search_term = request.args.get('q', '')
        
    # VULNERABLE SQL QUERY
    query = f"SELECT * FROM products WHERE name LIKE '%{search_term}%' OR description LIKE '%{search_term}%'"
        
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            conn.close()
                        
            return jsonify({
                'success': True,
                'results': results,
                'count': len(results),
                'query': query
            })
        except Exception as e:
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}',
                'query': query
            }), 500
    return jsonify({'success': False, 'message': 'Database error'}), 500

@app.route('/api/vulnerable/user/<user_id>', methods=['GET'])
def vulnerable_get_user(user_id):
    """Vulnerable user lookup"""
        
    # VULNERABLE SQL QUERY
    query = f"SELECT * FROM users WHERE id = {user_id}"
        
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            user = cursor.fetchone()
            cursor.close()
            conn.close()
                        
            return jsonify({
                'success': user is not None,
                'user': user,
                'query': query
            })
        except Exception as e:
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}',
                'query': query
            }), 500
    return jsonify({'success': False, 'message': 'Database error'}), 500

# ============= PROTECTED ENDPOINTS =============

@app.route('/api/protected/login', methods=['POST'])
def protected_login():
    """Protected login with SQLi detection"""
    if not detector:
        return jsonify({'success': False, 'message': 'Detector is not running.'}), 500

    username = request.form.get('username', '')
    password = request.form.get('password', '')
        
    # Check for SQL injection on *both* fields
    detection_user = detector.detect(username)
    detection_pass = detector.detect(password)
        
    if detection_user['is_malicious'] or detection_pass['is_malicious']:
        return jsonify({
            'success': False,
            'message': 'SQL Injection detected and blocked!',
            'detection': detection_user if detection_user['is_malicious'] else detection_pass
        }), 403
        
    # Safe parameterized query
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
        
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # This is safe. The driver handles escaping.
            cursor.execute(query, (username, password)) 
            user = cursor.fetchone()
            cursor.close()
            conn.close()
                        
            return jsonify({
                'success': user is not None,
                'message': 'Login successful' if user else 'Invalid credentials',
                'user': user if user else None
            })
        except Exception as e:
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    return jsonify({'success': False, 'message': 'Database error'}), 500

@app.route('/api/protected/search', methods=['GET'])
def protected_search():
    """Protected product search"""
    if not detector:
        return jsonify({'success': False, 'message': 'Detector is not running.'}), 500

    search_term = request.args.get('q', '')
        
    # Check for SQL injection
    detection = detector.detect(search_term)
        
    if detection['is_malicious']:
        return jsonify({
            'success': False,
            'message': 'SQL Injection detected and blocked!',
            'detection': detection
        }), 403
        
    # Safe parameterized query
    query = "SELECT * FROM products WHERE name LIKE %s OR description LIKE %s"
        
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True) # <-- [FIX 3] Was 'conn.cursor()'
            search_pattern = f'%{search_term}%'
            cursor.execute(query, (search_pattern, search_pattern))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
                        
            return jsonify({
                'success': True,
                'results': results,
                'count': len(results)
            })
        except Exception as e:
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    return jsonify({'success': False, 'message': 'Database error'}), 500

# ============= API ENDPOINTS (for dashboard/testing) =============

@app.route('/api/detect', methods=['POST'])
def detect_sqli():
    """API endpoint for SQL injection detection"""
    if not detector:
        return jsonify({'error': 'Detector is not running.'}), 500
        
    data = request.get_json()
    query = data.get('query', '')
        
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    result = detector.detect(query)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detection statistics"""
    if not detector:
        return jsonify({'error': 'Detector is not running.'}), 500
        
    stats = detector.get_statistics()
    return jsonify(stats)

if __name__ == '__main__':
    print("Starting SQLi Detection Demo Application...")
    print(f"Loading detector from: {MIDDLEWARE_PATH}")
    print(f"Connecting to database: {DB_CONFIG['host']}/{DB_CONFIG['database']}")
    if not detector:
        print("\n!!! WARNING: ML DETECTOR FAILED TO LOAD !!!")
        print("!!! The application will run, but protected endpoints will fail. !!!\n")
    print("Vulnerable endpoints: /api/vulnerable/*")
    print("Protected endpoints: /api/protected/*")
    # Running on port 5001 to match the proxy's default backend URL
    app.run(host='0.0.0.0', port=5001, debug=True)