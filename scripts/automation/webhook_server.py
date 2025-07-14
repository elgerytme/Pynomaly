#!/usr/bin/env python3
"""
GitHub Webhook Server for Real-time Issue Sync

This server handles GitHub webhook events and triggers automatic
synchronization of issues to TODO.md.
"""

import hashlib
import hmac
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sync_github_issues_to_todo import SyncManager, get_github_token

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
REPO_OWNER = "elgerytme"
REPO_NAME = "Pynomaly"
TODO_FILE = Path(__file__).parent.parent.parent / "TODO.md"
WEBHOOK_SECRET = os.getenv('GITHUB_WEBHOOK_SECRET')

# Global sync manager
sync_manager = None

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature"""
    if not WEBHOOK_SECRET:
        logger.warning("No webhook secret configured - skipping signature verification")
        return True
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle GitHub webhook events"""
    try:
        # Get signature
        signature = request.headers.get('X-Hub-Signature-256')
        if not signature:
            logger.error("Missing webhook signature")
            return jsonify({'error': 'Missing signature'}), 401
        
        # Verify signature
        payload = request.get_data()
        if not verify_webhook_signature(payload, signature):
            logger.error("Invalid webhook signature")
            return jsonify({'error': 'Invalid signature'}), 401
        
        # Parse webhook data
        event_type = request.headers.get('X-GitHub-Event')
        webhook_data = request.get_json()
        
        logger.info(f"Received {event_type} webhook")
        
        # Handle issue events
        if event_type == 'issues':
            action = webhook_data.get('action')
            issue_number = webhook_data.get('issue', {}).get('number')
            
            logger.info(f"Issue #{issue_number} {action}")
            
            # Trigger sync for relevant actions
            if action in ['opened', 'edited', 'closed', 'reopened', 'labeled', 'unlabeled']:
                success = sync_manager.handle_webhook(webhook_data)
                
                if success:
                    logger.info("Issue sync completed successfully")
                    return jsonify({'status': 'success', 'message': 'TODO.md updated'})
                else:
                    logger.error("Issue sync failed")
                    return jsonify({'status': 'error', 'message': 'Sync failed'}), 500
        
        # Handle issue comment events
        elif event_type == 'issue_comment':
            action = webhook_data.get('action')
            issue_number = webhook_data.get('issue', {}).get('number')
            
            logger.info(f"Issue #{issue_number} comment {action}")
            
            # Trigger sync for comment events that might affect priority/status
            if action in ['created', 'edited', 'deleted']:
                success = sync_manager.handle_webhook(webhook_data)
                
                if success:
                    logger.info("Comment sync completed successfully")
                    return jsonify({'status': 'success', 'message': 'TODO.md updated'})
                else:
                    logger.error("Comment sync failed")
                    return jsonify({'status': 'error', 'message': 'Sync failed'}), 500
        
        # For other events, just acknowledge
        return jsonify({'status': 'acknowledged'})
        
    except Exception as e:
        logger.error(f"Webhook handling error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'GitHub Issues Sync Webhook',
        'repo': f"{REPO_OWNER}/{REPO_NAME}",
        'todo_file': str(TODO_FILE)
    })

@app.route('/sync', methods=['POST'])
def manual_sync():
    """Manual sync endpoint"""
    try:
        success = sync_manager.sync_issues_to_todo()
        
        if success:
            return jsonify({'status': 'success', 'message': 'Manual sync completed'})
        else:
            return jsonify({'status': 'error', 'message': 'Manual sync failed'}), 500
            
    except Exception as e:
        logger.error(f"Manual sync error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def initialize_sync_manager():
    """Initialize the global sync manager"""
    global sync_manager
    
    token = get_github_token()
    sync_manager = SyncManager(REPO_OWNER, REPO_NAME, TODO_FILE, token)
    
    logger.info(f"Initialized sync manager for {REPO_OWNER}/{REPO_NAME}")

def main():
    """Main entry point"""
    # Initialize sync manager
    initialize_sync_manager()
    
    # Get port from environment
    port = int(os.getenv('PORT', 8080))
    
    logger.info(f"Starting webhook server on port {port}")
    logger.info(f"Webhook URL: http://localhost:{port}/webhook")
    logger.info(f"Health check: http://localhost:{port}/health")
    logger.info(f"Manual sync: http://localhost:{port}/sync")
    
    # Run server
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main()