from flask import Flask, request, jsonify, Response
import requests
# [FIX 1] Use relative imports for better package handling
from .detector import SQLiDetectorMiddleware
from .logger import ProxyLogger
import json
# [FIX 2] Removed unused 'import re'

app = Flask(__name__)

# Initialize detector and logger
detector = SQLiDetectorMiddleware()
logger = ProxyLogger()

# Configuration
BACKEND_URL = "http://localhost:5001"  # Backend database API
BLOCK_MALICIOUS = True
THRESHOLD = 0.7

def extract_sql_from_request(request_data):
    """Extract potential SQL queries from request"""
    queries = []
        
    # Check query parameters
    for key, value in request_data.args.items():
        if value:
            queries.append(str(value))
            
    # [FIX 3] Use 'request_data.method'. 'request_method' was undefined.
    # Also check for PUT requests, as they can contain form data.
    if request_data.method == 'POST' or request_data.method == 'PUT':
        # Check form data
        if request_data.form:
            for key, value in request_data.form.items():
                if value:
                    queries.append(str(value))
                    
        # Check JSON body
        if request_data.is_json:
            try:
                data = request_data.get_json()
                queries.extend(extract_from_dict(data))
            except Exception as e:
                logger.log_error({'error': 'Failed to parse JSON body', 'message': str(e)})

    # [FIX 4] CRITICAL: Unindented 'return queries'. 
    # It was inside the 'if request.is_json' block, causing the 
    # function to return 'None' for GET or form requests, which crashes the server.
    return queries

def extract_from_dict(d):
    """Recursively extract strings from dictionary/list"""
    strings = []
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, str):
                strings.append(value)
            elif isinstance(value, (dict, list)):
                strings.extend(extract_from_dict(value))
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, (dict, list)):
                strings.extend(extract_from_dict(item))
    return strings

# [FIX 5] Added 'PATCH' method for completeness
@app.route('/proxy/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy(path):
    """Proxy requests through SQLi detector"""
        
    # Extract potential SQL queries
    queries = extract_sql_from_request(request)
        
    # Detect malicious content
    malicious_detected = False
    detection_results = []
        
    if queries: # Only run detector if queries were found
        for query in queries:
            result = detector.detect(query, threshold=THRESHOLD)
            detection_results.append(result)
            if result['is_malicious']:
                malicious_detected = True
        
    # Log request
    log_entry = {
        'method': request.method,
        'path': path,
        'remote_addr': request.remote_addr,
        'queries_analyzed': len(queries),
        'malicious_detected': malicious_detected,
        'detection_results': detection_results
    }

    # Block if malicious
    if malicious_detected and BLOCK_MALICIOUS:
        logger.log_attack(log_entry) # Log as attack
        return jsonify({
            'error': 'SQL Injection detected',
            'message': 'Your request has been blocked due to suspicious content',
            'detection_results': detection_results
        }), 403
    
    # [FIX 6] Log as a normal request only if it was NOT blocked.
    logger.log_request(log_entry)
        
    # --- [FIX 7] ---
    # The original request forwarding was flawed. You cannot pass both
    # 'data' (for form) and 'json' (for json) to requests.post().
    # This is a much more robust method that forwards the raw request
    # exactly as it was received.
    try:
        url = f"{BACKEND_URL}/{path}"
        
        # Prepare request components
        headers = {key: value for (key, value) in request.headers if key != 'Host'}
        data = request.get_data() # Get raw body
        params = request.args     # Get query params
        cookies = request.cookies

        # Forward the request using the generic 'requests.request'
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=data,
            params=params,
            cookies=cookies,
            allow_redirects=False,
            stream=True # Use stream for large responses
        )

        # --- [FIX 8] ---
        # This is a more robust way to return the response to the client,
        # ensuring "hop-by-hop" headers (like 'Connection') are not forwarded.
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        resp_headers = [
            (key, value) for (key, value) in resp.raw.headers.items()
            if key.lower() not in excluded_headers
        ]

        return Response(resp.content, resp.status_code, resp_headers)

    except requests.exceptions.ConnectionError:
        error_msg = {'error': 'Backend service unavailable', 'url': BACKEND_URL}
        logger.log_error(error_msg)
        return jsonify(error_msg), 503
    except Exception as e:
        error_msg = {'error': 'Proxy error', 'message': str(e), 'path': path}
        logger.log_error(error_msg)
        return jsonify(error_msg), 500
    # --- END FIX 7 & 8 ---


@app.route('/proxy/stats', methods=['GET'])
def get_stats():
    """Get proxy statistics"""
    detector_stats = detector.get_statistics()
    logger_stats = logger.get_statistics()
        
    return jsonify({
        'detector': detector_stats,
        'logger': logger_stats
    })

@app.route('/proxy/logs', methods=['GET'])
def get_logs():
    """Get recent logs"""
    limit = request.args.get('limit', default=100, type=int)
    logs = logger.get_recent_logs(limit)
    return jsonify({'logs': logs})

@app.route('/proxy/attacks', methods=['GET'])
def get_attacks():
    """Get detected attacks"""
    limit = request.args.get('limit', default=100, type=int)
    attacks = logger.get_attacks(limit)
    return jsonify({'attacks': attacks})

@app.route('/proxy/config', methods=['GET', 'POST'])
def config():
    """Get/update proxy configuration"""
    global BLOCK_MALICIOUS, THRESHOLD
        
    if request.method == 'POST':
        data = request.get_json()
        if 'block_malicious' in data:
            BLOCK_MALICIOUS = bool(data['block_malicious'])
        if 'threshold' in data:
            THRESHOLD = float(data['threshold'])
                
        return jsonify({
            'message': 'Configuration updated',
            'config': {
                'block_malicious': BLOCK_MALICIOUS,
                'threshold': THRESHOLD
            }
        })
    else:
        return jsonify({
            'block_malicious': BLOCK_MALICIOUS,
            'threshold': THRESHOLD
        })

if __name__ == '__main__':
    print("Starting SQLi Detection Proxy...")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Block malicious: {BLOCK_MALICIOUS}")
    print(f"Detection threshold: {THRESHOLD}")
    # Set debug=False for production. 
    # debug=True is fine for testing but can be a security risk.
    app.run(host='0.0.0.0', port=5000, debug=True)