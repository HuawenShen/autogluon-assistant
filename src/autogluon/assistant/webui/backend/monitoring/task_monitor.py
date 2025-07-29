# src/autogluon/assistant/webui/backend/monitoring/task_monitor.py

import logging
from datetime import datetime
from typing import Dict, Optional

from .cloudwatch_client import CloudWatchClient

logger = logging.getLogger(__name__)


class TaskMonitor:
    """Monitor and log task execution"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.cloudwatch = CloudWatchClient()
        self._initialized = True
    
    def log_task_completion(self, run_id: str, task_info: Dict) -> None:
        """Log task completion to CloudWatch"""
        try:
            # 构建完整的任务记录
            task_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'run_id': run_id,
                'task_id': task_info.get('task_id'),
                'input_dir': task_info.get('input_dir'),
                'output_dir': task_info.get('output_dir'),
                'status': self._determine_status(task_info),
                'duration_seconds': self._calculate_duration(task_info),
                'start_time': task_info.get('start_time'),
                'end_time': datetime.utcnow().isoformat(),
                'cancelled': task_info.get('cancelled', False),
                'exit_code': task_info.get('exit_code'),
                'error_summary': self._extract_error_summary(task_info)
            }
            
            # 发送到CloudWatch
            self.cloudwatch.put_task_log(task_record)
            self.cloudwatch.put_metrics(task_record)
            
            logger.info(f"Task {run_id} logged to CloudWatch with status: {task_record['status']}")
            
        except Exception as e:
            logger.error(f"Failed to log task completion: {str(e)}", exc_info=True)
    
    def _determine_status(self, task_info: Dict) -> str:
        """确定任务最终状态"""
        if task_info.get('cancelled', False):
            return 'cancel'

        exit_code = task_info.get('exit_code')
        if exit_code == 0:
            return 'success'
        else:
            return 'fail'
    
    def _calculate_duration(self, task_info: Dict) -> Optional[float]:
        """计算任务执行时间"""
        start_time = task_info.get('start_time')
        if not start_time:
            return None
            
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.utcnow()
            return (end - start).total_seconds()
        except:
            return None
    
    def _extract_error_summary(self, task_info: Dict) -> Optional[str]:
        """提取错误摘要（如果有）"""
        if task_info.get('exit_code', 0) == 0:
            return None
            
        # 从日志中查找错误信息
        logs = task_info.get('logs', [])
        for log in reversed(logs[-10:]):  # 检查最后10条日志
            if 'error' in log.lower() or 'exception' in log.lower():
                return log[:200]  # 限制长度
                
        return "Unknown error"


# 全局实例
def get_task_monitor() -> TaskMonitor:
    """获取任务监控器实例"""
    return TaskMonitor()
