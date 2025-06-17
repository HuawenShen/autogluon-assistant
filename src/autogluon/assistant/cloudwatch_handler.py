# src/autogluon/assistant/cloudwatch_handler.py

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import boto3
from botocore.exceptions import ClientError


class CloudWatchHandler(logging.Handler):
    """
    Custom logging handler that sends WARNING and ERROR logs to CloudWatch Logs.
    Only active when running in WebUI environment.
    """
    
    def __init__(self, log_group_name: str = '/aws/autogluon-assistant/webui'):
        super().__init__()
        self.log_group_name = log_group_name
        self.log_stream_name = None
        self.logs_client = None
        self.sequence_token = None
        self.enabled = os.environ.get('AUTOGLUON_WEBUI', 'false').lower() == 'true'
        
        if self.enabled:
            try:
                self.logs_client = boto3.client('logs', region_name='us-west-2')
                self._ensure_log_group()
                self._create_log_stream()
                # Only handle WARNING and ERROR in CloudWatch
                self.setLevel(logging.WARNING)
            except Exception as e:
                # If CloudWatch setup fails, disable the handler
                self.enabled = False
                print(f"Failed to initialize CloudWatch handler: {str(e)}")
    
    def _ensure_log_group(self):
        """Ensure CloudWatch log group exists"""
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group_name)
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                raise
    
    def _create_log_stream(self):
        """Create a unique log stream for this process"""
        try:
            # Create unique stream name with timestamp and PID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pid = os.getpid()
            self.log_stream_name = f"process_{timestamp}_{pid}"
            
            self.logs_client.create_log_stream(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                raise
    
    def _get_sequence_token(self):
        """Get the current sequence token for the log stream"""
        try:
            response = self.logs_client.describe_log_streams(
                logGroupName=self.log_group_name,
                logStreamNamePrefix=self.log_stream_name,
                limit=1
            )
            
            if response['logStreams']:
                return response['logStreams'][0].get('uploadSequenceToken')
            return None
        except Exception:
            return None
    
    def emit(self, record):
        """
        Emit a log record to CloudWatch Logs.
        Only sends WARNING and ERROR level logs.
        """
        if not self.enabled or not self.logs_client or not self.log_stream_name:
            return
        
        # Only send WARNING and ERROR to CloudWatch
        if record.levelno < logging.WARNING:
            return
        
        try:
            # Format the log message
            msg = self.format(record)
            
            # Create structured log entry
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': msg,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                import traceback
                log_entry['exception'] = traceback.format_exception(*record.exc_info)
            
            # Prepare log event
            log_event = {
                'timestamp': int(record.created * 1000),
                'message': json.dumps(log_entry, ensure_ascii=False)
            }
            
            # Get current sequence token
            self.sequence_token = self._get_sequence_token()
            
            # Put log event
            put_params = {
                'logGroupName': self.log_group_name,
                'logStreamName': self.log_stream_name,
                'logEvents': [log_event]
            }
            
            if self.sequence_token:
                put_params['sequenceToken'] = self.sequence_token
            
            self.logs_client.put_log_events(**put_params)
            
        except Exception as e:
            # Don't propagate CloudWatch errors to avoid disrupting the application
            # Log to stderr for debugging
            import sys
            print(f"CloudWatch logging error: {str(e)}", file=sys.stderr)
