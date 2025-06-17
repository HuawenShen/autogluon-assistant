# src/autogluon/assistant/webui/backend/cloudwatch_manager.py

import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CloudWatchManager:
    """Manages CloudWatch metrics and logs for AutoGluon Assistant"""
    
    def __init__(self):
        """Initialize CloudWatch clients"""
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name='us-west-2')
            self.logs = boto3.client('logs', region_name='us-west-2')
            self.namespace = 'AutoGluonAssistant'
            self.log_group_name = '/aws/autogluon-assistant/webui'
            self.log_stream_name = None
            
            # Ensure log group exists
            self._ensure_log_group()
            
            # Create log stream for this instance
            self._create_log_stream()
            
        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch Manager: {str(e)}")
            self.cloudwatch = None
            self.logs = None
    
    def _ensure_log_group(self):
        """Ensure CloudWatch log group exists"""
        try:
            self.logs.create_log_group(logGroupName=self.log_group_name)
            logger.info(f"Created log group: {self.log_group_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.debug(f"Log group already exists: {self.log_group_name}")
            else:
                logger.error(f"Error creating log group: {str(e)}")
                raise
    
    def _create_log_stream(self):
        """Create a unique log stream for this instance"""
        try:
            # Use instance ID and timestamp for unique stream name
            instance_id = boto3.Session().region_name  # You can use EC2 instance ID if available
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_stream_name = f"webui_{timestamp}_{instance_id}"
            
            self.logs.create_log_stream(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name
            )
            logger.info(f"Created log stream: {self.log_stream_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.debug(f"Log stream already exists: {self.log_stream_name}")
            else:
                logger.error(f"Error creating log stream: {str(e)}")
                self.log_stream_name = None
    
    def send_task_started_metrics(self, run_id: str, task_config: Dict):
        """Send metrics when a task starts"""
        if not self.cloudwatch:
            return
        
        try:
            metrics = [
                {
                    'MetricName': 'TaskStarted',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Provider', 'Value': task_config.get('provider', 'unknown')},
                        {'Name': 'Model', 'Value': task_config.get('model', 'unknown')},
                        {'Name': 'ManualPrompts', 'Value': str(task_config.get('control', False))},
                        {'Name': 'MaxIterations', 'Value': str(task_config.get('max_iter', 0))}
                    ]
                },
                {
                    'MetricName': 'RunningTasks',
                    'Value': 1,
                    'Unit': 'Count'
                }
            ]
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics
            )
            
            logger.info(f"Sent task started metrics for {run_id}")
            
        except Exception as e:
            logger.error(f"Error sending task started metrics: {str(e)}")
    
    def send_task_completed_metrics(self, run_id: str, task_info: Dict):
        """Send metrics when a task completes"""
        if not self.cloudwatch:
            return
        
        try:
            status = task_info.get('status', 'unknown')  # success, failed, cancelled
            runtime_seconds = task_info.get('runtime_seconds', 0)
            token_usage = task_info.get('token_usage', {})
            
            metrics = [
                {
                    'MetricName': 'TaskCompleted',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Status', 'Value': status},
                        {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                        {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                    ]
                },
                {
                    'MetricName': 'RunningTasks',
                    'Value': -1,  # Decrement running tasks
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'TaskRuntime',
                    'Value': runtime_seconds,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                        {'Name': 'Status', 'Value': status}
                    ]
                }
            ]
            
            # Add token usage metrics if available
            if token_usage:
                token_metrics = [
                    {
                        'MetricName': 'TokensInput',
                        'Value': token_usage.get('total_input_tokens', 0),
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                            {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                        ]
                    },
                    {
                        'MetricName': 'TokensOutput',
                        'Value': token_usage.get('total_output_tokens', 0),
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                            {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                        ]
                    },
                    {
                        'MetricName': 'TokensTotal',
                        'Value': token_usage.get('total_tokens', 0),
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                            {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                        ]
                    }
                ]
                metrics.extend(token_metrics)
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics
            )
            
            logger.info(f"Sent task completed metrics for {run_id}")
            
        except Exception as e:
            logger.error(f"Error sending task completed metrics: {str(e)}")
    
    def send_log_event(self, level: str, message: str, run_id: Optional[str] = None):
        """Send a log event to CloudWatch Logs"""
        if not self.logs or not self.log_stream_name:
            return
        
        try:
            timestamp = int(time.time() * 1000)
            log_message = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message
            }
            
            if run_id:
                log_message['run_id'] = run_id
            
            # Get sequence token
            response = self.logs.describe_log_streams(
                logGroupName=self.log_group_name,
                logStreamNamePrefix=self.log_stream_name
            )
            
            sequence_token = None
            if response['logStreams']:
                sequence_token = response['logStreams'][0].get('uploadSequenceToken')
            
            # Put log events
            put_params = {
                'logGroupName': self.log_group_name,
                'logStreamName': self.log_stream_name,
                'logEvents': [
                    {
                        'timestamp': timestamp,
                        'message': json.dumps(log_message)
                    }
                ]
            }
            
            if sequence_token:
                put_params['sequenceToken'] = sequence_token
            
            self.logs.put_log_events(**put_params)
            
        except Exception as e:
            # Don't log error to avoid infinite loop
            pass
    
    def create_dashboard(self):
        """Create CloudWatch Dashboard with predefined widgets"""
        if not self.cloudwatch:
            return
        
        try:
            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskStarted", { "stat": "Sum", "period": 300 } ],
                                [ ".", "TaskCompleted", { "stat": "Sum", "period": 300 } ]
                            ],
                            "view": "timeSeries",
                            "stacked": False,
                            "region": "us-west-2",
                            "title": "Task Activity Over Time",
                            "period": 300
                        }
                    },
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskCompleted", "Status", "success", { "stat": "Sum" } ],
                                [ "...", "failed", { "stat": "Sum" } ],
                                [ "...", "cancelled", { "stat": "Sum" } ]
                            ],
                            "view": "pie",
                            "region": "us-west-2",
                            "title": "Task Success Rate",
                            "period": 86400
                        }
                    },
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TokensTotal", { "stat": "Sum", "period": 3600 } ]
                            ],
                            "view": "timeSeries",
                            "region": "us-west-2",
                            "title": "Token Usage Over Time",
                            "period": 300
                        }
                    },
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskRuntime", { "stat": "Average", "period": 3600 } ]
                            ],
                            "view": "timeSeries",
                            "region": "us-west-2",
                            "title": "Average Task Runtime",
                            "period": 300,
                            "yAxis": {
                                "left": {
                                    "label": "Seconds"
                                }
                            }
                        }
                    },
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 12,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "RunningTasks", { "stat": "Maximum", "period": 60 } ]
                            ],
                            "view": "timeSeries",
                            "region": "us-west-2",
                            "title": "Currently Running Tasks",
                            "period": 300
                        }
                    },
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 12,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TokensTotal", "Provider", "bedrock", { "stat": "Sum" } ],
                                [ "...", "openai", { "stat": "Sum" } ],
                                [ "...", "anthropic", { "stat": "Sum" } ]
                            ],
                            "view": "pie",
                            "region": "us-west-2",
                            "title": "Token Usage by Provider",
                            "period": 86400
                        }
                    }
                ]
            }
            
            self.cloudwatch.put_dashboard(
                DashboardName='AutoGluonAssistant-Dashboard',
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info("Created CloudWatch Dashboard: AutoGluonAssistant-Dashboard")
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")


# Global instance
_cloudwatch_manager = None


def get_cloudwatch_manager() -> CloudWatchManager:
    """Get or create the global CloudWatch manager instance"""
    global _cloudwatch_manager
    if _cloudwatch_manager is None:
        _cloudwatch_manager = CloudWatchManager()
    return _cloudwatch_manager