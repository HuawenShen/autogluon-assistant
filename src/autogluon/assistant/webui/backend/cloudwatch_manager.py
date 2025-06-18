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
            
            # Track running tasks count in memory
            self.running_tasks_count = 0
            
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            import os
            pid = os.getpid()
            self.log_stream_name = f"webui_{timestamp}_{pid}"
            
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
            # Increment running tasks count
            self.running_tasks_count += 1
            
            metrics = [
                {
                    'MetricName': 'TaskStarted',
                    'Value': 1.0,  # Use float
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow(),
                    'Dimensions': [
                        {'Name': 'Provider', 'Value': task_config.get('provider', 'unknown')},
                        {'Name': 'Model', 'Value': task_config.get('model', 'unknown')},
                        {'Name': 'ManualPrompts', 'Value': str(task_config.get('control', False))},
                        {'Name': 'MaxIterations', 'Value': str(task_config.get('max_iter', 5))}
                    ]
                },
                {
                    'MetricName': 'RunningTasks',
                    'Value': float(self.running_tasks_count),  # Current count as float
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow()
                }
            ]
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics
            )
            
            # Log task start event
            self.send_log_event('INFO', f'Task started: {run_id} with provider={task_config.get("provider", "unknown")}, model={task_config.get("model", "unknown")}', run_id)
            
            logger.info(f"Sent task started metrics for {run_id}")
            
        except Exception as e:
            logger.error(f"Error sending task started metrics: {str(e)}")
    
    def send_task_completed_metrics(self, run_id: str, task_info: Dict):
        """Send metrics when a task completes"""
        if not self.cloudwatch:
            return
        
        try:
            # Decrement running tasks count
            self.running_tasks_count = max(0, self.running_tasks_count - 1)
            
            status = task_info.get('status', 'unknown')  # success, failed, cancelled
            runtime_seconds = task_info.get('runtime_seconds', 0)
            token_usage = task_info.get('token_usage', {})
            
            metrics = [
                {
                    'MetricName': 'TaskCompleted',
                    'Value': 1.0,  # Use float
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow(),
                    'Dimensions': [
                        {'Name': 'Status', 'Value': status},
                        {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                        {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                    ]
                },
                {
                    'MetricName': 'RunningTasks',
                    'Value': float(self.running_tasks_count),  # Current count as float
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow()
                }
            ]
            
            # Add runtime metric only if greater than 0
            if runtime_seconds > 0:
                metrics.append({
                    'MetricName': 'TaskRuntime',
                    'Value': float(runtime_seconds),  # Use float
                    'Unit': 'Seconds',
                    'Timestamp': datetime.utcnow(),
                    'Dimensions': [
                        {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                        {'Name': 'Status', 'Value': status}
                    ]
                })
            
            # Add token usage metrics if available
            if token_usage:
                total_input = token_usage.get('total_input_tokens', 0)
                total_output = token_usage.get('total_output_tokens', 0)
                total_tokens = token_usage.get('total_tokens', 0)
                
                if total_input > 0:
                    metrics.append({
                        'MetricName': 'TokensInput',
                        'Value': float(total_input),  # Use float
                        'Unit': 'Count',
                        'Timestamp': datetime.utcnow(),
                        'Dimensions': [
                            {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                            {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                        ]
                    })
                
                if total_output > 0:
                    metrics.append({
                        'MetricName': 'TokensOutput',
                        'Value': float(total_output),  # Use float
                        'Unit': 'Count',
                        'Timestamp': datetime.utcnow(),
                        'Dimensions': [
                            {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                            {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                        ]
                    })
                
                if total_tokens > 0:
                    metrics.append({
                        'MetricName': 'TokensTotal',
                        'Value': float(total_tokens),  # Use float
                        'Unit': 'Count',
                        'Timestamp': datetime.utcnow(),
                        'Dimensions': [
                            {'Name': 'Provider', 'Value': task_info.get('provider', 'unknown')},
                            {'Name': 'Model', 'Value': task_info.get('model', 'unknown')}
                        ]
                    })
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics
            )
            
            # Log task completion event
            log_message = f'Task completed: {run_id} with status={status}, runtime={runtime_seconds}s'
            if token_usage:
                log_message += f', tokens={total_tokens}'
            self.send_log_event('INFO', log_message, run_id)
            
            logger.info(f"Sent task completed metrics for {run_id}")
            
        except Exception as e:
            logger.error(f"Error sending task completed metrics: {str(e)}")
    
    def send_log_event(self, level: str, message: str, run_id: Optional[str] = None):
        """Send a log event to CloudWatch Logs (only WARNING and ERROR levels)"""
        if not self.logs or not self.log_stream_name:
            return
        
        # Only send WARNING and ERROR logs to CloudWatch
        if level not in ['WARNING', 'ERROR']:
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
                    # Task Activity Over Time
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskStarted", { "stat": "Sum", "period": 300, "label": "Started" } ],
                                [ ".", "TaskCompleted", { "stat": "Sum", "period": 300, "label": "Completed" } ]
                            ],
                            "view": "timeSeries",
                            "stacked": False,
                            "region": "us-west-2",
                            "title": "Task Activity Over Time",
                            "period": 300,
                            "yAxis": {
                                "left": {
                                    "min": 0
                                }
                            }
                        }
                    },
                    # Total Tasks Summary
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskStarted", { "stat": "Sum", "period": 2592000, "label": "Total Started" } ],
                                [ ".", "TaskCompleted", { "stat": "Sum", "period": 2592000, "label": "Total Completed" } ]
                            ],
                            "view": "singleValue",
                            "region": "us-west-2",
                            "title": "Total Tasks (30 days)",
                            "period": 300
                        }
                    },
                    # Token Usage Over Time
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TokensTotal", { "stat": "Sum", "period": 300, "label": "Total Tokens" } ],
                                [ ".", "TokensInput", { "stat": "Sum", "period": 300, "label": "Input Tokens", "visible": False } ],
                                [ ".", "TokensOutput", { "stat": "Sum", "period": 300, "label": "Output Tokens", "visible": False } ]
                            ],
                            "view": "timeSeries",
                            "stacked": False,
                            "region": "us-west-2",
                            "title": "Token Usage Over Time",
                            "period": 300,
                            "yAxis": {
                                "left": {
                                    "min": 0
                                }
                            }
                        }
                    },
                    # Average Task Runtime
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskRuntime", { "stat": "Average", "period": 300, "label": "Average Runtime (seconds)" } ]
                            ],
                            "view": "timeSeries",
                            "stacked": False,
                            "region": "us-west-2",
                            "title": "Average Task Runtime",
                            "period": 300,
                            "yAxis": {
                                "left": {
                                    "min": 0,
                                    "label": "Seconds"
                                }
                            }
                        }
                    },
                    # Currently Running Tasks
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 12,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "RunningTasks", { "stat": "Maximum", "period": 60, "label": "Running Tasks" } ]
                            ],
                            "view": "timeSeries",
                            "region": "us-west-2",
                            "title": "Currently Running Tasks",
                            "period": 300,
                            "yAxis": {
                                "left": {
                                    "min": 0
                                }
                            }
                        }
                    },
                    # Task Success Rate (Pie Chart)
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 12,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskCompleted", "Status", "success", { "stat": "Sum", "period": 86400 } ],
                                [ "...", "failed", { "stat": "Sum", "period": 86400 } ],
                                [ "...", "cancelled", { "stat": "Sum", "period": 86400 } ]
                            ],
                            "view": "pie",
                            "region": "us-west-2",
                            "title": "Task Success Rate (24h)",
                            "period": 300,
                            "setPeriodToTimeRange": True
                        }
                    },
                    # Token Usage by Provider
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 18,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TokensTotal", "Provider", "bedrock", { "stat": "Sum", "period": 86400 } ],
                                [ "...", "openai", { "stat": "Sum", "period": 86400 } ],
                                [ "...", "anthropic", { "stat": "Sum", "period": 86400 } ]
                            ],
                            "view": "pie",
                            "region": "us-west-2",
                            "title": "Token Usage by Provider (24h)",
                            "period": 300,
                            "setPeriodToTimeRange": True
                        }
                    },
                    # Task Count by Model
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 18,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [ self.namespace, "TaskStarted", { "stat": "Sum", "period": 300 } ]
                            ],
                            "view": "bar",
                            "region": "us-west-2",
                            "title": "Tasks by Time Period",
                            "period": 300,
                            "yAxis": {
                                "left": {
                                    "min": 0
                                }
                            }
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