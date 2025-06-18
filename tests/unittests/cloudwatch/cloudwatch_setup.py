#!/usr/bin/env python3
"""
CloudWatch Test Script - Simulates actual WebUI task flow
This script mimics the exact way the WebUI sends metrics to CloudWatch
"""

import sys
import time
import json
import uuid
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parents[4]  # Navigate up to autogluon-assistant root
sys.path.insert(0, str(project_root / 'src'))

# Import the actual CloudWatchManager used by WebUI
from autogluon.assistant.webui.backend.cloudwatch_manager import get_cloudwatch_manager


def simulate_successful_task():
    """Simulate a complete successful task lifecycle"""
    
    print("=== Simulating Successful Task ===")
    
    # Get CloudWatch manager instance (same as WebUI does)
    cloudwatch = get_cloudwatch_manager()
    
    # Generate task ID (similar to WebUI)
    run_id = uuid.uuid4().hex
    print(f"Task ID: {run_id[:8]}...")
    
    # Task configuration (similar to what WebUI sends)
    task_config = {
        'provider': 'openai',
        'model': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'control': False,  # No manual prompts
        'max_iter': 6
    }
    
    print("\n1. Starting task...")
    # Send task started metrics (exactly as in utils.py start_run)
    cloudwatch.send_task_started_metrics(run_id, task_config)
    print("   ✓ Sent TaskStarted and RunningTasks metrics")
    
    # Simulate task running for a while
    print("\n2. Task running...")
    start_time = time.time()
    time.sleep(3)  # Simulate some work
    
    # Simulate token usage (similar to actual token_usage.json)
    token_usage = {
        'total_input_tokens': 22360,
        'total_output_tokens': 3845,
        'total_tokens': 26205
    }
    
    # Calculate runtime
    runtime_seconds = time.time() - start_time
    
    print("\n3. Completing task...")
    # Send task completed metrics (exactly as in utils.py)
    task_info = {
        'status': 'success',
        'runtime_seconds': runtime_seconds,
        'token_usage': token_usage,
        'provider': task_config['provider'],
        'model': task_config['model']
    }
    
    cloudwatch.send_task_completed_metrics(run_id, task_info)
    print("   ✓ Sent TaskCompleted, TaskRuntime, TokensTotal, and updated RunningTasks")
    
    print(f"\n✅ Task simulation complete!")
    print(f"   - Runtime: {runtime_seconds:.1f} seconds")
    print(f"   - Tokens used: {token_usage['total_tokens']}")


def simulate_failed_task():
    """Simulate a failed task"""
    
    print("\n=== Simulating Failed Task ===")
    
    cloudwatch = get_cloudwatch_manager()
    run_id = uuid.uuid4().hex
    print(f"Task ID: {run_id[:8]}...")
    
    task_config = {
        'provider': 'openai',
        'model': 'gpt-4o-2024-08-06',
        'control': True,  # Manual prompts enabled
        'max_iter': 3
    }
    
    print("\n1. Starting task...")
    cloudwatch.send_task_started_metrics(run_id, task_config)
    print("   ✓ Sent TaskStarted and RunningTasks metrics")
    
    start_time = time.time()
    time.sleep(2)  # Simulate some work before failure
    runtime_seconds = time.time() - start_time
    
    print("\n2. Task failed!")
    # Send failure metrics (no token usage for failed task)
    task_info = {
        'status': 'failed',
        'runtime_seconds': runtime_seconds,
        'token_usage': {},  # Empty for failed task
        'provider': task_config['provider'],
        'model': task_config['model']
    }
    
    cloudwatch.send_task_completed_metrics(run_id, task_info)
    print("   ✓ Sent TaskCompleted with failed status")


def simulate_cancelled_task():
    """Simulate a cancelled task"""
    
    print("\n=== Simulating Cancelled Task ===")
    
    cloudwatch = get_cloudwatch_manager()
    run_id = uuid.uuid4().hex
    print(f"Task ID: {run_id[:8]}...")
    
    task_config = {
        'provider': 'anthropic',
        'model': 'claude-3-7-sonnet-20250219',
        'control': False,
        'max_iter': 10
    }
    
    print("\n1. Starting task...")
    cloudwatch.send_task_started_metrics(run_id, task_config)
    print("   ✓ Sent TaskStarted and RunningTasks metrics")
    
    start_time = time.time()
    time.sleep(1)  # Simulate brief work before cancel
    runtime_seconds = time.time() - start_time
    
    print("\n2. Task cancelled by user")
    # Send cancellation metrics
    task_info = {
        'status': 'cancelled',
        'runtime_seconds': runtime_seconds,
        'token_usage': {},
        'provider': task_config['provider'],
        'model': task_config['model']
    }
    
    cloudwatch.send_task_completed_metrics(run_id, task_info)
    print("   ✓ Sent TaskCompleted with cancelled status")


def simulate_multiple_concurrent_tasks():
    """Simulate multiple tasks running concurrently"""
    
    print("\n=== Simulating Multiple Concurrent Tasks ===")
    
    cloudwatch = get_cloudwatch_manager()
    
    # Start 3 tasks
    tasks = []
    for i in range(3):
        run_id = uuid.uuid4().hex
        task_config = {
            'provider': 'bedrock',
            'model': 'claude-3-sonnet',
            'control': False,
            'max_iter': 5
        }
        
        print(f"\nStarting task {i+1} ({run_id[:8]}...)...")
        cloudwatch.send_task_started_metrics(run_id, task_config)
        
        tasks.append({
            'run_id': run_id,
            'config': task_config,
            'start_time': time.time()
        })
        
        time.sleep(0.5)  # Stagger starts
    
    print(f"\n✓ {len(tasks)} tasks running concurrently")
    print(f"  RunningTasks should show: {cloudwatch.running_tasks_count}")
    
    # Complete tasks one by one
    for i, task in enumerate(tasks):
        time.sleep(1)  # Simulate work
        
        runtime = time.time() - task['start_time']
        token_usage = {
            'total_input_tokens': 10000 + i * 5000,
            'total_output_tokens': 2000 + i * 1000,
            'total_tokens': 12000 + i * 6000
        }
        
        task_info = {
            'status': 'success',
            'runtime_seconds': runtime,
            'token_usage': token_usage,
            'provider': task['config']['provider'],
            'model': task['config']['model']
        }
        
        print(f"\nCompleting task {i+1} ({task['run_id'][:8]}...)...")
        cloudwatch.send_task_completed_metrics(task['run_id'], task_info)
        print(f"  RunningTasks should show: {cloudwatch.running_tasks_count}")


def verify_metrics():
    """Print instructions for verifying metrics"""
    
    print("\n" + "="*60)
    print("VERIFICATION STEPS:")
    print("="*60)
    print("\n1. Wait 1-2 minutes for metrics to appear in CloudWatch")
    print("\n2. Check the dashboard:")
    print(f"   https://console.aws.amazon.com/cloudwatch/home?region=us-west-2#dashboards:name=AutoGluonAssistant-Dashboard")
    print("\n3. Expected results:")
    print("   - Task Activity: Should show Started and Completed lines")
    print("   - Total Tasks: Should show count of started/completed")
    print("   - Token Usage: Should show token consumption over time")
    print("   - Average Runtime: Should show runtime in seconds")
    print("   - Running Tasks: Should show 0 (all tasks completed)")
    print("   - Success Rate: Should show pie chart with success/failed/cancelled")
    print("\n4. Check raw metrics:")
    print("   - Go to CloudWatch > Metrics > AutoGluonAssistant")
    print("   - Verify all metric names have proper Unit values")


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'simple':
            # Just one successful task
            simulate_successful_task()
        elif command == 'all':
            # All scenarios
            simulate_successful_task()
            time.sleep(2)
            simulate_failed_task()
            time.sleep(2)
            simulate_cancelled_task()
            time.sleep(2)
            simulate_multiple_concurrent_tasks()
        elif command == 'concurrent':
            # Just concurrent tasks
            simulate_multiple_concurrent_tasks()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python cloudwatch_setup.py [simple|all|concurrent]")
            sys.exit(1)
    else:
        # Default: simulate one of each type
        print("Simulating various task scenarios...\n")
        simulate_successful_task()
        time.sleep(2)
        simulate_failed_task()
        time.sleep(2)
        simulate_cancelled_task()
    
    verify_metrics()


if __name__ == "__main__":
    main()