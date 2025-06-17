# src/autogluon/assistant/webui/backend/__init__.py

# Import CloudWatch manager to ensure it's available
from .cloudwatch_manager import get_cloudwatch_manager

__all__ = ['get_cloudwatch_manager']
