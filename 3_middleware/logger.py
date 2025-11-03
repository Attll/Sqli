import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from collections import deque
import time

class ProxyLogger:
    """
    Handles logging for requests, attacks, and errors.
    """
    
    def __init__(self, log_dir='logs', max_log_size=10485760, backup_count=5):
        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory history for quick stats
        self.recent_logs = deque(maxlen=200)
        self.recent_attacks = deque(maxlen=200)
        self.stats = {
            'total_requests': 0,
            'malicious_detected': 0,
            'errors': 0,
            'start_time': time.time()
        }

        # --- General Request Logger (for all traffic) ---
        self.request_logger = logging.getLogger('proxy_requests')
        self.request_logger.setLevel(logging.INFO)
        self.request_logger.propagate = False
        if not self.request_logger.hasHandlers():
            request_handler = RotatingFileHandler(
                self.log_dir / 'proxy.log', 
                maxBytes=max_log_size, 
                backupCount=backup_count
            )
            # Use JSON for easier parsing
            request_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "log": %(message)s}')
            request_handler.setFormatter(request_formatter)
            self.request_logger.addHandler(request_handler)

        # --- Attack Logger (for blocked requests) ---
        self.attack_logger = logging.getLogger('proxy_attacks')
        self.attack_logger.setLevel(logging.WARNING)
        self.attack_logger.propagate = False
        if not self.attack_logger.hasHandlers():
            attack_handler = RotatingFileHandler(
                self.log_dir / 'attacks.log', 
                maxBytes=max_log_size, 
                backupCount=backup_count
            )
            attack_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "log": %(message)s}')
            attack_handler.setFormatter(attack_formatter)
            self.attack_logger.addHandler(attack_handler)

    def _to_json(self, data):
        """Safely convert log data to a JSON string."""
        try:
            return json.dumps(data)
        except TypeError:
            return json.dumps(str(data)) # Fallback

    def log_request(self, log_entry):
        """Log a standard (non-malicious) request."""
        self.stats['total_requests'] += 1
        self.recent_logs.appendleft(log_entry)
        self.request_logger.info(self._to_json(log_entry))

    def log_attack(self, log_entry):
        """Log a detected attack."""
        # Also log to the main request log for a complete traffic view
        self.log_request(log_entry) 
        self.stats['malicious_detected'] += 1
        self.recent_attacks.appendleft(log_entry)
        self.attack_logger.warning(self._to_json(log_entry)) # Log to separate attack file

    def log_error(self, error_entry):
        """Log a proxy or backend error."""
        self.stats['errors'] += 1
        self.request_logger.error(self._to_json(error_entry))

    def get_statistics(self):
        """Get in-memory statistics."""
        self.stats['uptime_seconds'] = time.time() - self.stats['start_time']
        return self.stats

    def get_recent_logs(self, limit=100):
        """Get recent logs from in-memory deque."""
        return list(self.recent_logs)[:limit]

    def get_attacks(self, limit=100):
        """Get recent attacks from in-memory deque."""
        return list(self.recent_attacks)[:limit]