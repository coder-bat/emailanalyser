#!/usr/bin/env python3
"""
Flask API server for EmailAnalyser frontend
Provides REST API endpoints to serve email analysis data
"""
import os
import sys
import json
import csv
import re
import time
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, abort, send_file
from flask_cors import CORS
import logging
import threading
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

# Import security utilities
import sys
sys.path.insert(0, os.path.dirname(__file__))
from security_utils import SecurityUtils, SecureFilePath

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_frontend_build_dir = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
app = Flask(
    __name__,
    static_folder=_frontend_build_dir,
    static_url_path=''  # so /static maps to build/static automatically
)

# Configure CORS - restrict in production
def get_cors_origins():
    """Get allowed CORS origins based on environment."""
    if os.environ.get('FLASK_ENV') == 'production':
        # In production, use configured origins or default to same-origin
        origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:5000').split(',')
        return origins
    return '*'

CORS(app, resources={r"/api/*": {"origins": get_cors_origins()}})

# Configuration
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'email_analysis_output')
API_PORT = int(os.environ.get('API_PORT', 5000))
MAX_JOBS = int(os.environ.get('MAX_JOBS', '100'))
JOB_TTL_SECONDS = int(os.environ.get('JOB_TTL_SECONDS', '3600'))  # 1 hour TTL for completed jobs

# In-memory job tracking with size limit and TTL
jobs_lock = threading.Lock()
jobs = {}
job_access_times = {}  # Track last access time for TTL cleanup

def _sanitize_env_value(value):
    """Sanitize environment variable values to prevent command injection.
    
    DEPRECATED: Use SecurityUtils.sanitize_env_value() instead for consistent
    security handling across the codebase.
    """
    # Delegate to SecurityUtils for consistent validation
    return SecurityUtils.sanitize_env_value(value)

def _validate_email(email):
    """Basic email validation."""
    if not email:
        return False
    return SecurityUtils.validate_email(email)

def _validate_positive_int(value, default=None):
    """Validate and return a positive integer."""
    try:
        val = int(value)
        if val <= 0:
            return default
        return val
    except (ValueError, TypeError):
        return default

def _cleanup_old_jobs():
    """Clean up old completed/failed jobs to prevent memory leaks using TTL."""
    with jobs_lock:
        current_time = time.time()
        jobs_to_remove = []
        
        # Remove old completed/failed jobs based on TTL
        for job_id, job_info in jobs.items():
            status = job_info.get('status')
            updated_at = job_info.get('updated_at', 0)
            
            if status in ('completed', 'failed'):
                if current_time - updated_at > JOB_TTL_SECONDS:
                    jobs_to_remove.append(job_id)
        
        # Also enforce MAX_JOBS limit if needed
        if len(jobs) > MAX_JOBS:
            # Sort by last update time
            sorted_jobs = sorted(jobs.items(), key=lambda x: x[1].get('updated_at', 0))
            # Remove oldest jobs that are completed or failed
            for job_id, job_info in sorted_jobs:
                if job_info.get('status') in ('completed', 'failed') and job_id not in jobs_to_remove:
                    jobs_to_remove.append(job_id)
                if len(jobs) - len(jobs_to_remove) <= MAX_JOBS:
                    break
        
        # Actually remove the jobs
        for job_id in jobs_to_remove:
            if job_id in jobs:
                del jobs[job_id]
            if job_id in job_access_times:
                del job_access_times[job_id]

def _run_analysis_job(job_id: str, params: dict):
    """Worker thread to execute main.py analysis and update job status."""
    import time
    
    # Validate params is a dict
    if not isinstance(params, dict):
        logger.error(f"Job {job_id}: params must be a dictionary")
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].update({
                    'status': 'failed',
                    'error': 'Invalid parameters: params must be a dictionary',
                    'updated_at': time.time()
                })
        return
    
    # Validate job exists before starting
    with jobs_lock:
        if job_id not in jobs:
            logger.error(f"Job {job_id} not found in jobs dictionary")
            return
        # Atomic update of job status
        jobs[job_id].update({
            'status': 'running',
            'progress': 5,
            'updated_at': time.time()
        })
        job_access_times[job_id] = time.time()
    
    # Use a local variable to track if we've cleaned up to avoid double cleanup
    cleanup_done = False
    
    env = os.environ.copy()
    # Pass selected params as env vars understood by main.py with sanitization
    if params.get('email'):
        if not _validate_email(params['email']):
            with jobs_lock:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = 'Invalid email address format'
                jobs[job_id]['updated_at'] = time.time()
            return
        env['EMAIL_ADDRESS'] = SecurityUtils.sanitize_env_value(params['email'])
    
    # Validate max_emails with upper bound check
    is_valid, max_emails = SecurityUtils.validate_max_emails(params.get('max_emails', 1000))
    if not is_valid:
        logger.warning(f"Invalid max_emails value, using sanitized value: {max_emails}")
    env['MAX_EMAILS'] = str(max_emails)
    
    if params.get('categories'):
        # Validate categories - only allow alphanumeric and comma
        categories = str(params['categories'])
        if SecurityUtils.validate_categories(categories):
            env['GMAIL_CATEGORIES'] = categories[:SecurityUtils.MAX_CATEGORIES_LENGTH]
        else:
            logger.warning(f"Invalid categories format rejected: {categories[:50]}")
    
    if params.get('unread_only'):
        env['EMAIL_SEARCH_CRITERIA'] = 'UNSEEN'
    
    if params.get('password'):
        # Password is passed securely via env var, limited length
        password = params['password']
        # Additional validation: check password length before sanitization
        if len(password) > SecurityUtils.MAX_PASSWORD_LENGTH:
            logger.warning(f"Password exceeds max length, truncating to {SecurityUtils.MAX_PASSWORD_LENGTH} chars")
        # Use SecurityUtils for consistent sanitization
        env['EMAIL_PASSWORD'] = SecurityUtils.sanitize_password(password)
    
    try:
        # Determine Python executable - prefer python3 for compatibility
        python_exe = sys.executable if sys.executable else 'python3'
        
        # Use list instead of string to avoid shell injection
        proc = subprocess.Popen(
            [python_exe, 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        # Stream output to log and attempt crude progress updates
        for line in proc.stdout:  # type: ignore
            stripped = line.rstrip()
            logger.info(f"[job {job_id}] {stripped}")
            lower = stripped.lower()
            progress = None
            # Stage markers
            if '[1/7]' in stripped:
                progress = 5
            elif '[2/7]' in stripped:
                progress = 15
            elif '[3/7]' in stripped:
                progress = 28
            # Header batch / body fetch hints
            elif 'performing batched header fetch' in lower:
                progress = 32
            elif 'header fetch completed' in lower:
                progress = 38
            elif 'will fetch full bodies for' in lower:
                progress = 42
            elif 'retrieved ' in lower and ' emails' in lower:
                progress = 48
            elif '[4/7]' in stripped:
                progress = 58
            elif '[5/7]' in stripped:
                progress = 70
            elif '[6/7]' in stripped:
                progress = 83
            elif '[7/7]' in stripped:
                progress = 92
            elif 'analysis complete' in lower:
                progress = 96
            if progress is not None:
                with jobs_lock:
                    if jobs.get(job_id):
                        # Only increase (never regress)
                        if progress > jobs[job_id].get('progress', 0):
                            jobs[job_id]['progress'] = progress
                            jobs[job_id]['updated_at'] = time.time()
        rc = proc.wait()
        with jobs_lock:
            if jobs.get(job_id):
                jobs[job_id].update({
                    'status': 'completed' if rc == 0 else 'failed',
                    'progress': 100 if rc == 0 else jobs[job_id].get('progress', 90),
                    'return_code': rc,
                    'updated_at': time.time()
                })
        cleanup_done = True
        _cleanup_old_jobs()
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        with jobs_lock:
            if jobs.get(job_id):
                jobs[job_id].update({
                    'status': 'failed',
                    'error': str(e),
                    'updated_at': time.time()
                })
        if not cleanup_done:
            _cleanup_old_jobs()


# Simple in-memory cache for API responses
response_cache = {}
response_cache_lock = threading.Lock()
CACHE_TTL = int(os.environ.get('CACHE_TTL', '60'))  # Default 60 seconds

def _get_cached_response(cache_key):
    """Get cached response if not expired."""
    with response_cache_lock:
        if cache_key in response_cache:
            data, timestamp = response_cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                return data
            else:
                del response_cache[cache_key]
    return None

def _set_cached_response(cache_key, data):
    """Cache response with timestamp."""
    with response_cache_lock:
        response_cache[cache_key] = (data, time.time())

def get_latest_file(pattern):
    """Find the most recent file matching pattern in output directory"""
    # Sanitize pattern to prevent path traversal
    if not pattern or not SecurityUtils.validate_path_pattern(pattern):
        logger.warning(f"Invalid file pattern rejected: {pattern}")
        return None
    
    if not os.path.exists(OUTPUT_DIR):
        return None
    
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if pattern in f]
        if not files:
            return None
        
        # Sort by modification time, newest first
        # Handle potential errors during stat
        def get_mtime_safe(filename):
            try:
                return os.path.getmtime(os.path.join(OUTPUT_DIR, filename))
            except (OSError, ValueError) as e:
                logger.debug(f"Could not get mtime for {filename}: {e}")
                return 0
        
        files.sort(key=get_mtime_safe, reverse=True)
        
        # Ensure we have a valid file after sorting
        if not files or get_mtime_safe(files[0]) == 0:
            return None
            
        result = os.path.join(OUTPUT_DIR, files[0])
        
        # Double-check the result is within OUTPUT_DIR (prevent traversal)
        result_abs = os.path.abspath(result)
        output_abs = os.path.abspath(OUTPUT_DIR)
        if not result_abs.startswith(output_abs + os.sep) and result_abs != output_abs:
            logger.warning(f"Path traversal detected: {result}")
            return None
        
        return result
    except (OSError, PermissionError) as e:
        logger.error(f"Error accessing output directory: {e}")
        return None

def read_csv_file(filepath):
    """Read CSV file and return data as list of dictionaries"""
    if not filepath or not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error(f"Error reading CSV file {filepath}: {e}")
        return []

def read_json_file(filepath):
    """Read JSON file and return data"""
    if not filepath or not os.path.exists(filepath):
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {filepath}: {e}")
        return {}

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get analysis summary"""
    try:
        # Check cache first
        cache_key = 'summary'
        cached = _get_cached_response(cache_key)
        if cached:
            return jsonify(cached)
        
        # Read summary.json
        summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
        summary_data = read_json_file(summary_path)
        
        # Read email data to get additional stats
        email_data_path = get_latest_file('email_data_')
        email_data = read_csv_file(email_data_path)
        
        # Calculate additional metrics
        total_emails = len(email_data)
        important_emails = len([e for e in email_data if float(e.get('importance_score', 0)) >= 0.6])
        
        # Get date range
        dates = [e.get('date', '') for e in email_data if e.get('date')]
        date_range = {
            'start': min(dates) if dates else '',
            'end': max(dates) if dates else ''
        }
        
        response = {
            'total_senders': summary_data.get('total_senders', 0),
            'senders_to_delete_count': summary_data.get('senders_to_delete_count', 0),
            'important_senders_count': summary_data.get('important_senders_count', 0),
            'total_emails': total_emails,
            'important_emails': important_emails,
            'date_range': date_range
        }
        
        # Cache the response
        _set_cached_response(cache_key, response)
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/emails', methods=['GET'])
def get_emails():
    """Get email data with safe type conversion"""
    try:
        limit = request.args.get('limit', type=int)
        email_data_path = get_latest_file('email_data_')
        email_data = read_csv_file(email_data_path)
        
        # Convert data types and format with error handling
        processed_emails = []
        for email in email_data:
            try:
                processed_email = {
                    **email,
                    'importance_score': float(email.get('importance_score', 0) or 0),
                    'has_attachments': str(email.get('has_attachments', '')).lower() in ('true', '1', 'yes', 'on')
                }
                processed_emails.append(processed_email)
            except (ValueError, TypeError) as e:
                logger.debug(f"Error processing email data: {e}")
                # Include raw data if conversion fails
                processed_emails.append(email)
        
        if limit and limit > 0:
            processed_emails = processed_emails[:limit]
        
        return jsonify(processed_emails)
    except Exception as e:
        logger.error(f"Error getting emails: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sender-stats', methods=['GET'])
def get_sender_stats():
    """Get sender statistics with safe type conversion"""
    try:
        sender_stats_path = os.path.join(OUTPUT_DIR, 'sender_stats.csv')
        sender_data = read_csv_file(sender_stats_path)
        
        # Convert data types with error handling
        processed_senders = []
        for sender in sender_data:
            try:
                processed_sender = {
                    **sender,
                    'total_emails': int(sender.get('total_emails', 0) or 0),
                    'important_emails': int(sender.get('important_emails', 0) or 0)
                }
                processed_senders.append(processed_sender)
            except (ValueError, TypeError) as e:
                logger.debug(f"Error processing sender data: {e}")
                processed_senders.append(sender)
        
        return jsonify(processed_senders)
    except Exception as e:
        logger.error(f"Error getting sender stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/senders-to-delete', methods=['GET'])
def get_senders_to_delete():
    """Get senders recommended for deletion with safe type conversion"""
    try:
        delete_path = os.path.join(OUTPUT_DIR, 'senders_to_delete.csv')
        delete_data = read_csv_file(delete_path)
        
        # Convert data types with error handling
        processed_senders = []
        for sender in delete_data:
            try:
                processed_sender = {
                    **sender,
                    'count': int(sender.get('count', 0) or 0),
                    'avg_importance': float(sender.get('avg_importance', 0) or 0),
                    'newsletter_pct': float(sender.get('newsletter_pct', 0) or 0)
                }
                processed_senders.append(processed_sender)
            except (ValueError, TypeError) as e:
                logger.debug(f"Error processing sender to delete: {e}")
                processed_senders.append(sender)
        
        return jsonify(processed_senders)
    except Exception as e:
        logger.error(f"Error getting senders to delete: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/important-senders', methods=['GET'])
def get_important_senders():
    """Get important senders with safe type conversion"""
    try:
        important_path = os.path.join(OUTPUT_DIR, 'important_senders.csv')
        important_data = read_csv_file(important_path)
        
        # Convert data types with error handling
        processed_senders = []
        for sender in important_data:
            try:
                processed_sender = {
                    **sender,
                    'count': int(sender.get('count', 0) or 0),
                    'avg_importance': float(sender.get('avg_importance', 0) or 0)
                }
                processed_senders.append(processed_sender)
            except (ValueError, TypeError) as e:
                logger.debug(f"Error processing important sender: {e}")
                processed_senders.append(sender)
        
        return jsonify(processed_senders)
    except Exception as e:
        logger.error(f"Error getting important senders: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patterns', methods=['GET'])
def get_patterns():
    """Get analysis patterns (mock data for now)"""
    try:
        # Read email data to generate patterns
        email_data_path = get_latest_file('email_data_')
        email_data = read_csv_file(email_data_path)
        
        # Generate category distribution
        categories = {}
        for email in email_data:
            cat = email.get('category', 'other')
            categories[cat] = categories.get(cat, 0) + 1
        
        # Generate time patterns (simplified)
        hour_distribution = {}
        day_distribution = {}
        
        for email in email_data:
            try:
                dt = datetime.fromisoformat(email.get('date', ''))
                hour = dt.hour
                day = dt.strftime('%A')
                hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
                day_distribution[day] = day_distribution.get(day, 0) + 1
            except:
                continue
        
        # Generate sender patterns
        sender_counts = {}
        for email in email_data:
            sender = email.get('sender', '')
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        
        top_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        patterns = {
            'time_patterns': {
                'peak_hours': sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:3],
                'peak_days': sorted(day_distribution.items(), key=lambda x: x[1], reverse=True)[:3],
                'hour_distribution': hour_distribution,
                'day_distribution': day_distribution
            },
            'sender_patterns': {
                'top_senders': top_senders,
                'sender_categories': {}
            },
            'category_distribution': categories,
            'attachment_patterns': {
                'percentage_with_attachments': 25.0,
                'common_types': [['pdf', 15], ['jpg', 10], ['doc', 8]]
            }
        }
        
        return jsonify(patterns)
    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/run-analysis', methods=['POST'])
def run_analysis():
    try:
        params = request.get_json(force=True, silent=True) or {}
        
        # Validate params is a dict
        if not isinstance(params, dict):
            return jsonify({'error': 'Invalid request body'}), 400
        
        # Validate max_emails early to prevent DoS
        is_valid, max_emails = SecurityUtils.validate_max_emails(params.get('max_emails', 1000))
        if not is_valid:
            logger.warning(f"Invalid max_emails from client, using: {max_emails}")
        params['max_emails'] = max_emails
        
        # Rate limiting: check if too many jobs are queued/running
        with jobs_lock:
            active_count = sum(1 for info in jobs.values() if info.get('status') in ('queued', 'running'))
            if active_count >= 3:  # Max 3 concurrent jobs
                return jsonify({'error': 'Too many active jobs. Please wait for existing jobs to complete.'}), 429
        
        # Validate email format if provided
        if params.get('email') and not _validate_email(params['email']):
            return jsonify({'error': 'Invalid email address format'}), 400
        
        # Validate categories format if provided
        if params.get('categories'):
            if not SecurityUtils.validate_categories(str(params['categories'])):
                return jsonify({'error': 'Invalid categories format'}), 400
        
        # Simple single-active-job guard: if a job is already running or queued, reuse it
        reuse_job_id = None
        with jobs_lock:
            for jid, info in jobs.items():
                if info.get('status') in ('queued', 'running'):
                    reuse_job_id = jid
                    break
            if reuse_job_id:
                logger.info(f"Reusing active job {reuse_job_id} instead of starting a new one")
                return jsonify({'message': 'Analysis already in progress', 'job_id': reuse_job_id, 'active': True}), 202

            job_id = f"job_{uuid.uuid4().hex[:8]}"
            jobs[job_id] = {
                'status': 'queued',
                'progress': 0,
                'params': {k: params.get(k) for k in ('email','max_emails','categories','unread_only') if k != 'password'},  # do not expose password back
                'created_at': time.time(),
                'updated_at': time.time()
            }
            job_access_times[job_id] = time.time()
        t = threading.Thread(target=_run_analysis_job, args=(job_id, params), daemon=True)
        t.start()
        return jsonify({'message': 'Analysis started', 'job_id': job_id, 'active': False})
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analysis-status/<job_id>', methods=['GET'])
def get_analysis_status(job_id):
    # Validate job_id format to prevent injection
    if not SecurityUtils.validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID format'}), 400
    
    with jobs_lock:
        info = jobs.get(job_id)
        if info:
            job_access_times[job_id] = time.time()
    if not info:
        return jsonify({'error': 'job not found'}), 404
    return jsonify({k: v for k, v in info.items() if k in ('status','progress','error','return_code','params')})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Serve React frontend in production
@app.route('/')
def serve_root():  # explicit root
    index_path = os.path.join(_frontend_build_dir, 'index.html')
    if not os.path.exists(index_path):
        logger.error(f"index.html not found in {_frontend_build_dir}")
        abort(500)
    return send_file(index_path)

@app.errorhandler(404)
def spa_fallback(e):
    """Single Page App fallback: serve index.html for non-API 404s."""
    req_path = request.path
    if req_path.startswith('/api') or req_path.startswith('/health'):
        return jsonify({'error': 'Not found'}), 404
    index_path = os.path.join(_frontend_build_dir, 'index.html')
    if os.path.exists(index_path):
        logger.debug(f"SPA fallback for path: {req_path}")
        return send_file(index_path)
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create some sample data if none exists
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    if not os.path.exists(summary_path):
        sample_summary = {
            'total_senders': 25,
            'senders_to_delete_count': 5,
            'important_senders_count': 8
        }
        with open(summary_path, 'w') as f:
            json.dump(sample_summary, f, indent=2)
    
    # Determine debug mode from environment
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
    
    logger.info(f"Starting EmailAnalyser API server on port {API_PORT}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Debug mode: {debug_mode}")
    
    # Never run with debug=True in production
    if os.environ.get('FLASK_ENV') == 'production':
        debug_mode = False
    
    app.run(host='0.0.0.0', port=API_PORT, debug=debug_mode)
