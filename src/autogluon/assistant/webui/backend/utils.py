# src/autogluon/assistant/webui/backend/utils.py

import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from autogluon.assistant.constants import WEBUI_INPUT_MARKER, WEBUI_INPUT_REQUEST, WEBUI_OUTPUT_DIR

# Import CloudWatch manager
from .cloudwatch_manager import get_cloudwatch_manager

# Setup logging - reduce verbosity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence watchdog debug logs
logging.getLogger("watchdog").setLevel(logging.WARNING)

# Global storage for each run's state
_runs: Dict[str, Dict] = {}


def parse_log_line(line: str) -> dict:
    """
    Parse a log line according to format "<LEVEL> <content>".
    Also detect special WebUI input requests and output directory.

    Returns:
        {
            "level": "<BRIEF/INFO/MODEL_INFO or other>",
            "text": "<content text>",
            "special": "<type of special message if any>"
        }
    """
    # Skip empty lines
    if not line.strip():
        return None

    # Check for special WebUI output directory
    if line.strip().startswith(WEBUI_OUTPUT_DIR):
        output_dir = line.strip()[len(WEBUI_OUTPUT_DIR) :].strip()
        return {"level": "OUTPUT_DIR", "text": output_dir, "special": "output_dir"}

    # Check for special WebUI input request
    if line.strip().startswith(WEBUI_INPUT_REQUEST):
        prompt = line.strip()[len(WEBUI_INPUT_REQUEST) :].strip()
        return {"level": "INPUT_REQUEST", "text": prompt, "special": "input_request"}

    # Regular log parsing
    allowed_levels = {"ERROR", "BRIEF", "INFO", "DETAIL", "DEBUG", "WARNING"}
    stripped = line.strip()

    parts = stripped.split(" ", 1)
    if len(parts) == 2 and parts[0] in allowed_levels:
        # Skip empty BRIEF logs
        if parts[0] == "BRIEF" and not parts[1].strip():
            return None
        
        # Send WARNING and ERROR logs to CloudWatch
        if parts[0] in ["WARNING", "ERROR"]:
            # Get current run_id if available
            current_run_id = None
            for run_id, run_info in _runs.items():
                if not run_info.get("finished", True):
                    current_run_id = run_id
                    break
            
            cloudwatch = get_cloudwatch_manager()
            cloudwatch.send_log_event(parts[0], parts[1], current_run_id)
        
        return {"level": parts[0], "text": parts[1]}
    else:
        return {"level": "other", "text": stripped}


def start_run(run_id: str, cmd: List[str], credentials: Optional[Dict[str, str]] = None):
    """
    Start subprocess with stdin/stdout/stderr pipes.
    Set AUTOGLUON_WEBUI environment variable to indicate WebUI environment.
    Optionally set credentials (AWS, OpenAI, Anthropic) if provided.
    """
    # Extract task configuration from command
    task_config = {
        'provider': 'unknown',
        'model': 'unknown',
        'control': '--need-user-input' in cmd,
        'max_iter': 5  # default
    }
    
    # Parse max iterations from command
    for i, arg in enumerate(cmd):
        if arg == '-n' and i + 1 < len(cmd):
            try:
                task_config['max_iter'] = int(cmd[i + 1])
            except:
                pass
    
    # Parse provider and model from credentials
    if credentials:
        # Check for model info passed from routes
        if '_model_provider' in credentials:
            task_config['provider'] = credentials.pop('_model_provider')
        if '_model_name' in credentials:
            task_config['model'] = credentials.pop('_model_name')
        
        # If not already set, infer from API keys
        if task_config['provider'] == 'unknown':
            if 'AWS_ACCESS_KEY_ID' in credentials:
                task_config['provider'] = 'bedrock'
            elif 'OPENAI_API_KEY' in credentials:
                task_config['provider'] = 'openai'
            elif 'ANTHROPIC_API_KEY' in credentials:
                task_config['provider'] = 'anthropic'
    
    _runs[run_id] = {
        "process": None,
        "logs": [],
        "pointer": 0,
        "finished": False,
        "waiting_for_input": False,
        "input_prompt": None,
        "output_dir": None,
        "lock": threading.Lock(),
        "start_time": time.time(),
        "task_config": task_config,
    }
    
    # Send task started metrics
    cloudwatch = get_cloudwatch_manager()
    cloudwatch.send_task_started_metrics(run_id, task_config)

    def _target():
        try:
            # Set environment variable to indicate WebUI
            env = os.environ.copy()
            env["AUTOGLUON_WEBUI"] = "true"

            # Set credentials if provided
            if credentials:
                logger.info(f"Setting credentials for task {run_id[:8]}...")

                # Apply all provided environment variables
                for key, value in credentials.items():
                    env[key] = value
                    # Log environment variables (mask sensitive values)
                    if "KEY" in key or "TOKEN" in key:
                        masked_value = value[:4] + "..." if len(value) > 4 else "***"
                        logger.info(f"Task {run_id[:8]}: Setting {key}={masked_value}")
                    else:
                        logger.info(f"Task {run_id[:8]}: Setting {key}={value}")

                # Log which type of credentials were set based on what's actually present
                if "AWS_ACCESS_KEY_ID" in credentials:
                    logger.info(f"Task {run_id[:8]}: AWS credentials configured")
                    if _runs[run_id]["task_config"]["provider"] == "unknown":
                        _runs[run_id]["task_config"]["provider"] = "bedrock"
                if "OPENAI_API_KEY" in credentials:
                    logger.info(f"Task {run_id[:8]}: OpenAI API key configured")
                    if _runs[run_id]["task_config"]["provider"] == "unknown":
                        _runs[run_id]["task_config"]["provider"] = "openai"
                if "ANTHROPIC_API_KEY" in credentials:
                    logger.info(f"Task {run_id[:8]}: Anthropic API key configured")
                    if _runs[run_id]["task_config"]["provider"] == "unknown":
                        _runs[run_id]["task_config"]["provider"] = "anthropic"
            else:
                logger.info(f"Task {run_id[:8]}: No credentials provided, using system defaults")

            # Log the command being executed for debugging
            logger.info(f"Task {run_id[:8]}: Executing command: {' '.join(cmd)}")

            # Create process with stdin pipe
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,  # Enable stdin pipe
                text=True,
                env=env,
                bufsize=1,  # Line buffered
                # Create new process group for proper termination
                preexec_fn=os.setsid if os.name != "nt" else None,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            _runs[run_id]["process"] = p

            logger.info(f"Started task {run_id[:8]}...")

            # Read stdout line by line
            for line in p.stdout:
                line = line.rstrip("\n")

                with _runs[run_id]["lock"]:
                    # Parse the line
                    parsed = parse_log_line(line)

                    # Skip None results (empty lines, etc.)
                    if parsed is None:
                        continue

                    # Check if this is output directory notification
                    if parsed.get("special") == "output_dir":
                        _runs[run_id]["output_dir"] = parsed["text"]
                        logger.info(f"Task {run_id[:8]} output directory: {parsed['text']}")
                        # Don't add this to logs
                        continue

                    # Check if this is an input request
                    if parsed.get("special") == "input_request":
                        _runs[run_id]["waiting_for_input"] = True
                        _runs[run_id]["input_prompt"] = parsed["text"]
                        logger.info(f"Task {run_id[:8]} requesting user input")

                    # Always append to logs (original line, not parsed)
                    _runs[run_id]["logs"].append(line)

            p.wait()
            exit_code = p.returncode
            logger.info(f"Task {run_id[:8]} completed with exit code {exit_code}")
            
            # Determine task status
            status = "failed" if exit_code != 0 else "success"
            
            # Load token usage if available
            token_usage = {}
            output_dir = _runs[run_id].get("output_dir")
            if output_dir:
                token_file = Path(output_dir) / "token_usage.json"
                if token_file.exists():
                    try:
                        with open(token_file, 'r') as f:
                            token_data = json.load(f)
                            token_usage = token_data.get('total', {})
                    except Exception as e:
                        logger.error(f"Error loading token usage: {str(e)}")
            
            # Send completion metrics
            runtime_seconds = time.time() - _runs[run_id]["start_time"]
            task_info = {
                'status': status,
                'runtime_seconds': runtime_seconds,
                'token_usage': token_usage,
                'provider': _runs[run_id]["task_config"]["provider"],
                'model': _runs[run_id]["task_config"]["model"]
            }
            
            cloudwatch = get_cloudwatch_manager()
            cloudwatch.send_task_completed_metrics(run_id, task_info)

        except Exception as e:
            logger.error(f"Error in task {run_id[:8]}: {str(e)}", exc_info=True)
            with _runs[run_id]["lock"]:
                _runs[run_id]["logs"].append(f"Process error: {str(e)}")
            
            # Send failure metrics
            runtime_seconds = time.time() - _runs[run_id]["start_time"]
            task_info = {
                'status': 'failed',
                'runtime_seconds': runtime_seconds,
                'token_usage': {},
                'provider': _runs[run_id]["task_config"]["provider"],
                'model': _runs[run_id]["task_config"]["model"]
            }
            
            cloudwatch = get_cloudwatch_manager()
            cloudwatch.send_task_completed_metrics(run_id, task_info)
        finally:
            with _runs[run_id]["lock"]:
                _runs[run_id]["finished"] = True
                _runs[run_id]["waiting_for_input"] = False

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()


def send_user_input(run_id: str, user_input: str) -> bool:
    """
    Send user input to the subprocess stdin.
    Returns True if successful, False otherwise.
    """
    info = _runs.get(run_id)
    if not info:
        logger.error(f"Run {run_id} not found")
        return False

    with info["lock"]:
        process = info.get("process")
        if not process or not process.stdin or process.poll() is not None:
            logger.error(f"Process not available for input: {run_id}")
            return False

        try:
            # Send input with special marker
            input_line = f"{WEBUI_INPUT_MARKER}{user_input}\n"
            process.stdin.write(input_line)
            process.stdin.flush()

            # Reset input waiting state
            info["waiting_for_input"] = False
            info["input_prompt"] = None

            # Log the user input for display with proper formatting
            if user_input:
                info["logs"].append(f"BRIEF User input: {user_input}")
            else:
                info["logs"].append("BRIEF User input: (skipped)")

            logger.info(f"Sent input to task {run_id[:8]}")
            return True

        except Exception as e:
            logger.error(f"Error sending input to task {run_id[:8]}: {str(e)}")
            return False


def get_logs(run_id: str) -> List[str]:
    """
    Return list of new log lines since last call.
    """
    info = _runs.get(run_id)
    if info is None:
        return []

    with info["lock"]:
        logs = info["logs"]
        ptr = info["pointer"]
        new = logs[ptr:]
        info["pointer"] = len(logs)
        return new


def get_status(run_id: str) -> dict:
    """
    Return task status including whether it's waiting for input and output directory.
    """
    info = _runs.get(run_id)
    if info is None:
        return {"finished": True, "error": "run_id not found"}

    with info["lock"]:
        return {
            "finished": info["finished"],
            "waiting_for_input": info.get("waiting_for_input", False),
            "input_prompt": info.get("input_prompt", None),
            "output_dir": info.get("output_dir", None),
        }


def cancel_run(run_id: str):
    """
    Terminate the subprocess for the given run.
    """
    info = _runs.get(run_id)
    if info and info["process"] and not info["finished"]:
        process = info["process"]

        try:
            if os.name == "nt":  # Windows
                process.terminate()
            else:  # Unix/Linux/Mac
                # Send SIGTERM to entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            # Give process time to exit gracefully
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()

            # Add cancellation log
            with info["lock"]:
                info["logs"].append("Task cancelled by user")
            
            # Send cancellation metrics
            runtime_seconds = time.time() - info["start_time"]
            task_info = {
                'status': 'cancelled',
                'runtime_seconds': runtime_seconds,
                'token_usage': {},
                'provider': info["task_config"]["provider"],
                'model': info["task_config"]["model"]
            }
            
            cloudwatch = get_cloudwatch_manager()
            cloudwatch.send_task_completed_metrics(run_id, task_info)

        except Exception as e:
            with info["lock"]:
                info["logs"].append(f"Error cancelling task: {str(e)}")
        finally:
            with info["lock"]:
                info["finished"] = True
                info["waiting_for_input"] = False
