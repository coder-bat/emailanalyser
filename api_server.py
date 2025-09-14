#!/usr/bin/env python3
"""
Flask API server for EmailAnalyser frontend
Provides REST API endpoints to serve email analysis data
"""
import os
import json
import csv
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'email_analysis_output')
API_PORT = int(os.environ.get('API_PORT', 5000))

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
    """Run email analysis (stub for now)"""
    try:
        config = request.get_json()
        # In a real implementation, this would trigger the main.py analysis
        # For now, return a success message
        return jsonify({
            'message': 'Analysis started successfully',
            'job_id': f'job_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        })
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis-status/<job_id>', methods=['GET'])
def get_analysis_status(job_id):
    """Get analysis status (stub for now)"""
    return jsonify({
        'status': 'completed',
        'progress': 100
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Serve React frontend in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend"""
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend', 'build')
    if path != "" and os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    else:
        return send_from_directory(frontend_dir, 'index.html')

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