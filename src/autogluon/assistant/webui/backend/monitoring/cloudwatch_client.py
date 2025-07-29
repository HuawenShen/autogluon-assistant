# src/autogluon/assistant/webui/backend/monitoring/cloudwatch_client.py

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CloudWatchClient:
    """CloudWatch client for logging and metrics"""
    
    def __init__(self):
        # 从环境变量读取配置
        self.region = "us-west-2"
        self.log_group = '/aws/autogluon-assistant/tasks'
        self.namespace = 'AutoGluonAssistant'
        
        # 初始化clients
        try:
            self.logs_client = boto3.client('logs', region_name=self.region)
            self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
            self.enabled = True
            logger.info("CloudWatch monitoring enabled")
        except Exception as e:
            logger.warning(f"CloudWatch monitoring disabled: {str(e)}")
            self.enabled = False
    
    def put_task_log(self, task_data: Dict) -> bool:
        """Send task completion log to CloudWatch"""
        if not self.enabled:
            return False
            
        try:
            # 使用run_id作为log stream name
            log_stream = f"task-{task_data['run_id']}"
            
            # 创建log stream（如果不存在）
            try:
                self.logs_client.create_log_stream(
                    logGroupName=self.log_group,
                    logStreamName=log_stream
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                    raise
            
            # 发送日志事件
            self.logs_client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=log_stream,
                logEvents=[
                    {
                        'timestamp': int(datetime.utcnow().timestamp() * 1000),
                        'message': json.dumps(task_data, default=str)
                    }
                ]
            )
            
            logger.info(f"Task log sent to CloudWatch: {task_data['run_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send log to CloudWatch: {str(e)}")
            return False
    
    def put_metrics(self, task_data: Dict) -> bool:
        """Send task metrics to CloudWatch"""
        if not self.enabled:
            return False
            
        try:
            metrics = []
            
            # 任务状态计数
            status = task_data['status']
            metrics.append({
                'MetricName': f'Task{status.title()}Count',
                'Value': 1,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Environment', 'Value': 'production'}
                ]
            })
            
            # 执行时间（如果有）
            if task_data.get('duration_seconds'):
                metrics.append({
                    'MetricName': 'TaskDuration',
                    'Value': task_data['duration_seconds'],
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Status', 'Value': status}
                    ]
                })
            
            # 发送metrics
            self.cloudwatch_client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send metrics to CloudWatch: {str(e)}")
            return False
