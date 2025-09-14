#!/usr/bin/env python3
"""
Flask API server for EmailAnalyser frontend
Provides REST API endpoints to serve email analysis data
"""
import os
import json
import csv
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, abort, send_file
from flask_cors import CORS
import logging
import threading
import subprocess
import uuid
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_frontend_build_dir = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
app = Flask(
    __name__,
    static_folder=_frontend_build_dir,
    static_url_path=''  # so /static maps to build/static automatically
)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable broad CORS for API endpoints

# Configuration
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'email_analysis_output')
API_PORT = int(os.environ.get('API_PORT', 5000))

# In-memory job tracking
jobs_lock = threading.Lock()
jobs = {}

def _run_analysis_job(job_id: str, params: dict):
    """Worker thread to execute main.py analysis and update job status."""
    with jobs_lock:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['progress'] = 5
    env = os.environ.copy()
    # Pass selected params as env vars understood by main.py
    if params.get('email'):
        env['EMAIL_ADDRESS'] = params['email']
    if params.get('max_emails'):
        env['MAX_EMAILS'] = str(params['max_emails'])
    if params.get('categories'):
        env['GMAIL_CATEGORIES'] = params['categories']
    if params.get('unread_only'):
        env['EMAIL_SEARCH_CRITERIA'] = 'UNSEEN'
    if params.get('password'):
        env['EMAIL_PASSWORD'] = params['password']
    try:
        proc = subprocess.Popen([
            'python', 'main.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        # Stream output to log and attempt crude progress updates
        for line in proc.stdout:  # type: ignore
            stripped = line.rstrip()
            logger.info(f"[job {job_id}] {stripped}")
            lower = stripped.lower()
            progress = None
            # Stage markers
            if '[1/7]' in stripped:
                progress = max(5, jobs[job_id].get('progress', 0))
            elif '[2/7]' in stripped:
                progress = 15
            elif '[3/7]' in stripped:
                progress = max(28, jobs[job_id].get('progress', 0))
            # Header batch / body fetch hints
            elif 'performing batched header fetch' in lower:
                progress = max(32, jobs[job_id].get('progress', 0))
            elif 'header fetch completed' in lower:
                progress = max(38, jobs[job_id].get('progress', 0))
            elif 'will fetch full bodies for' in lower:
                progress = max(42, jobs[job_id].get('progress', 0))
            elif 'retrieved ' in lower and ' emails' in lower:
                progress = max(48, jobs[job_id].get('progress', 0))
            elif '[4/7]' in stripped:
                progress = max(58, jobs[job_id].get('progress', 0))
            elif '[5/7]' in stripped:
                progress = max(70, jobs[job_id].get('progress', 0))
            elif '[6/7]' in stripped:
                progress = max(83, jobs[job_id].get('progress', 0))
            elif '[7/7]' in stripped:
                progress = max(92, jobs[job_id].get('progress', 0))
            elif 'analysis complete' in lower:
                progress = max(96, jobs[job_id].get('progress', 0))
            if progress is not None:
                with jobs_lock:
                    if jobs.get(job_id):
                        # Only increase (never regress)
                        if progress > jobs[job_id].get('progress', 0):
                            jobs[job_id]['progress'] = progress
        rc = proc.wait()
        with jobs_lock:
            if jobs.get(job_id):
                jobs[job_id]['status'] = 'completed' if rc == 0 else 'failed'
                jobs[job_id]['progress'] = 100 if rc == 0 else jobs[job_id].get('progress', 90)
                jobs[job_id]['return_code'] = rc
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        with jobs_lock:
            if jobs.get(job_id):
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = str(e)


def get_latest_file(pattern):
    """Find the most recent file matching pattern in output directory"""
    if not os.path.exists(OUTPUT_DIR):
        return None
    
    files = [f for f in os.listdir(OUTPUT_DIR) if pattern in f]
    if not files:
        return None
    
    # Sort by modification time, newest first
    files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    return os.path.join(OUTPUT_DIR, files[0])

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
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/emails', methods=['GET'])
def get_emails():
    """Get email data"""
    try:
        limit = request.args.get('limit', type=int)
        email_data_path = get_latest_file('email_data_')
        email_data = read_csv_file(email_data_path)
        
        # Convert data types and format
        for email in email_data:
            email['importance_score'] = float(email.get('importance_score', 0))
            email['has_attachments'] = email.get('has_attachments', 'False').lower() == 'true'
        
        if limit:
            email_data = email_data[:limit]
        
        return jsonify(email_data)
    except Exception as e:
        logger.error(f"Error getting emails: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sender-stats', methods=['GET'])
def get_sender_stats():
    """Get sender statistics"""
    try:
        sender_stats_path = os.path.join(OUTPUT_DIR, 'sender_stats.csv')
        sender_data = read_csv_file(sender_stats_path)
        
        # Convert data types
        for sender in sender_data:
            sender['total_emails'] = int(sender.get('total_emails', 0))
            sender['important_emails'] = int(sender.get('important_emails', 0))
        
        return jsonify(sender_data)
    except Exception as e:
        logger.error(f"Error getting sender stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/senders-to-delete', methods=['GET'])
def get_senders_to_delete():
    """Get senders recommended for deletion"""
    try:
        delete_path = os.path.join(OUTPUT_DIR, 'senders_to_delete.csv')
        delete_data = read_csv_file(delete_path)
        
        # Convert data types
        for sender in delete_data:
            sender['count'] = int(sender.get('count', 0))
            sender['avg_importance'] = float(sender.get('avg_importance', 0))
            sender['newsletter_pct'] = float(sender.get('newsletter_pct', 0))
        
        return jsonify(delete_data)
    except Exception as e:
        logger.error(f"Error getting senders to delete: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/important-senders', methods=['GET'])
def get_important_senders():
    """Get important senders"""
    try:
        important_path = os.path.join(OUTPUT_DIR, 'important_senders.csv')
        important_data = read_csv_file(important_path)
        
        # Convert data types
        for sender in important_data:
            sender['count'] = int(sender.get('count', 0))
            sender['avg_importance'] = float(sender.get('avg_importance', 0))
        
        return jsonify(important_data)
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
        params = request.get_json(force=True) or {}
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
                'params': {k: params.get(k) for k in ('email','max_emails','categories','unread_only','password') if k != 'password'}  # do not expose password back
            }
        t = threading.Thread(target=_run_analysis_job, args=(job_id, params), daemon=True)
        t.start()
        return jsonify({'message': 'Analysis started', 'job_id': job_id, 'active': False})
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis-status/<job_id>', methods=['GET'])
def get_analysis_status(job_id):
    with jobs_lock:
        info = jobs.get(job_id)
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
    
    logger.info(f"Starting EmailAnalyser API server on port {API_PORT}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    app.run(host='0.0.0.0', port=API_PORT, debug=True)