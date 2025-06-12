# src/autogluon/assistant/webui/pages/Run_dataset.py

import copy
import uuid
from datetime import datetime
from pathlib import Path
import time
import shutil
from typing import Dict, List, Optional, Any, Tuple
import requests
from dataclasses import dataclass, field

import streamlit as st

from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.webui.log_processor import messages, process_logs, render_task_logs
from autogluon.assistant.webui.result_manager import render_task_results
from autogluon.assistant.constants import INITIAL_STAGE, SUCCESS_MESSAGE, API_URL


# ==================== Constants ====================
PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

VERBOSITY_MAP = {
    "DETAIL": "3",
    "INFO": "2", 
    "BRIEF": "1",
}


# ==================== Data Classes ====================
@dataclass
class Message:
    """聊天消息"""
    role: str
    type: str
    content: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def text(cls, text: str, role: str = "assistant") -> "Message":
        return cls(role=role, type="text", content={"text": text})
    
    @classmethod
    def user_summary(cls, summary: str, input_dir: Optional[str] = None) -> "Message":
        content = {"summary": summary}
        if input_dir:
            content["input_dir"] = input_dir
        return cls(role="user", type="user_summary", content=content)
    
    @classmethod
    def command(cls, command: str) -> "Message":
        return cls(role="assistant", type="command", content={"command": command})
    
    @classmethod
    def task_log(cls, run_id: str, phase_states: Dict, max_iter: int, output_dir: Optional[str] = None, input_dir: Optional[str] = None) -> "Message":
        content = {
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase_states": phase_states,
            "max_iter": max_iter,
        }
        if output_dir:
            content["output_dir"] = output_dir
        if input_dir:
            content["input_dir"] = input_dir
        return cls(
            role="assistant", 
            type="task_log",
            content=content
        )
    
    @classmethod
    def task_results(cls, run_id: str, output_dir: str) -> "Message":
        return cls(
            role="assistant",
            type="task_results",
            content={
                "run_id": run_id,
                "output_dir": output_dir
            }
        )


@dataclass 
class TaskConfig:
    """任务配置"""
    uploaded_config: Any
    max_iter: int
    control: bool
    log_verbosity: str


# ==================== Session State ====================
class SessionState:
    """会话状态管理器"""
    
    @staticmethod
    def init():
        """初始化会话状态"""
        defaults = {
            "user_session_id": uuid.uuid4().hex,
            "messages": [Message.text("Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start.")],
            "data_src": None,
            "task_running": False,
            "run_id": None,
            "current_task_logs": [],
            "running_config": None,
            "current_input_dir": None,
            "waiting_for_input": False,
            "input_prompt": None,
            "current_iteration": 0,
            # 新增：控制 iteration 信息显示
            "show_iteration_info": False,
            "iteration_info_content": None,
            "clearing_iteration_info": False,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def start_task(run_id: str, config: TaskConfig, input_dir: str):
        """开始新任务"""
        st.session_state.task_running = True
        st.session_state.run_id = run_id
        st.session_state.current_task_logs = []
        st.session_state.running_config = config
        st.session_state.current_input_dir = input_dir
        st.session_state.waiting_for_input = False
        st.session_state.input_prompt = None
        st.session_state.current_iteration = 0
        st.session_state.show_iteration_info = False
        st.session_state.iteration_info_content = None
        
        # 清理旧的日志处理器
        SessionState._cleanup_processors()
    
    @staticmethod
    def finish_task():
        """结束任务"""
        st.session_state.task_running = False
        st.session_state.running_config = None
        st.session_state.current_task_logs = []
        st.session_state.current_input_dir = None
        st.session_state.waiting_for_input = False
        st.session_state.input_prompt = None
        st.session_state.current_iteration = 0
        st.session_state.show_iteration_info = False
        st.session_state.iteration_info_content = None
        st.session_state.clearing_iteration_info = False
        
        # 清理当前任务的处理器
        if st.session_state.run_id:
            processor_key = f"log_processor_{st.session_state.run_id}"
            if processor_key in st.session_state:
                del st.session_state[processor_key]
    
    @staticmethod
    def set_waiting_for_input(waiting: bool, prompt: Optional[str] = None, iteration: Optional[int] = None):
        """设置等待输入状态"""
        st.session_state.waiting_for_input = waiting
        st.session_state.input_prompt = prompt
        if iteration is not None:
            st.session_state.current_iteration = iteration
        
        # 如果开始等待输入，显示 iteration 信息
        if waiting:
            st.session_state.show_iteration_info = True
            st.session_state.clearing_iteration_info = False
    
    @staticmethod
    def clear_iteration_info():
        """清空 iteration 信息显示"""
        st.session_state.show_iteration_info = False
        st.session_state.iteration_info_content = None
        st.session_state.clearing_iteration_info = True
    
    @staticmethod
    def add_message(message: Message):
        """添加消息"""
        st.session_state.messages.append(message)
    
    @staticmethod
    def delete_task_from_history(run_id: str):
        """从历史中删除任务相关的消息"""
        # First, find the task_log message to get its index
        task_log_index = None
        for i, msg in enumerate(st.session_state.messages):
            if msg.type == "task_log" and msg.content.get("run_id") == run_id:
                task_log_index = i
                break
        
        if task_log_index is None:
            return
        
        # Find the associated messages (user_summary, command, iteration_prompts, etc.)
        start_index = task_log_index
        
        # Look backwards for related messages
        i = task_log_index - 1
        while i >= 0:
            msg = st.session_state.messages[i]
            
            # Check for various message types
            if msg.type in ["command", "user_summary"]:
                start_index = i
                i -= 1
                continue
            
            # Check for cancel-related messages
            elif msg.type == "text" and msg.role == "user" and msg.content.get("text", "").strip().lower() == "cancel":
                start_index = i
                i -= 1
                continue
            
            # Check for cancel confirmation message
            elif msg.type == "text" and msg.role == "assistant" and "has been cancelled" in msg.content.get("text", ""):
                start_index = i
                i -= 1
                continue
            
            # If we hit any other message type, stop looking
            else:
                break
        
        # Find the end index (task_results should be right after task_log)
        end_index = task_log_index
        if (task_log_index + 1 < len(st.session_state.messages) and 
            st.session_state.messages[task_log_index + 1].type == "task_results" and
            st.session_state.messages[task_log_index + 1].content.get("run_id") == run_id):
            end_index = task_log_index + 1
        
        # Create new message list without the task-related messages
        new_messages = []
        for i, msg in enumerate(st.session_state.messages):
            if i < start_index or i > end_index:
                new_messages.append(msg)
        
        st.session_state.messages = new_messages
    
    @staticmethod
    def _cleanup_processors():
        """清理旧的日志处理器"""
        keys_to_delete = [k for k in st.session_state if k.startswith("log_processor_")]
        for key in keys_to_delete:
            del st.session_state[key]


# ==================== Backend API ====================
class BackendAPI:
    """后端API通信"""
    
    @staticmethod
    def start_task(data_src: str, config_path: str, user_prompt: str, config: TaskConfig) -> str:
        """启动任务"""
        payload = {
            "data_src": data_src,
            "config_path": config_path,
            "max_iter": config.max_iter,
            "init_prompt": user_prompt or None,
            "control": config.control,
            "verbosity": VERBOSITY_MAP[config.log_verbosity],
        }
        
        response = requests.post(f"{API_URL}/run", json=payload)
        return response.json()["run_id"]
    
    @staticmethod
    def fetch_logs(run_id: str) -> List[Dict]:
        """获取日志"""
        response = requests.get(f"{API_URL}/logs", params={"run_id": run_id})
        return response.json().get("lines", [])
    
    @staticmethod
    def check_status(run_id: str) -> Dict:
        """检查任务状态"""
        response = requests.get(f"{API_URL}/status", params={"run_id": run_id})
        return response.json()
    
    @staticmethod
    def send_user_input(run_id: str, user_input: str) -> bool:
        """发送用户输入到后端"""
        try:
            response = requests.post(f"{API_URL}/input", json={
                "run_id": run_id,
                "input": user_input
            })
            return response.json().get("success", False)
        except Exception as e:
            st.error(f"Error sending input: {str(e)}")
            return False
    
    @staticmethod
    def cancel_task(run_id: str) -> bool:
        """取消任务"""
        try:
            response = requests.post(f"{API_URL}/cancel", json={"run_id": run_id})
            return response.json().get("cancelled", False)
        except:
            return False


# ==================== UI Components ====================
class UI:
    """UI组件"""
    
    @staticmethod
    def setup_page():
        """设置页面"""
        st.set_page_config(page_title="AutoMLAgent Chat", layout="wide")
    
    @staticmethod
    def render_sidebar() -> TaskConfig:
        """渲染侧边栏"""
        with st.sidebar:
            with st.expander("⚙️ Settings", expanded=False):
                config = TaskConfig(
                    uploaded_config=st.file_uploader(
                        "Config file (optional)",
                        type=['yaml', 'yml'],
                        key="config_uploader",
                        help="Upload a custom YAML config file. If not provided, default config will be used."
                    ),
                    max_iter=st.number_input("Max iterations", min_value=1, max_value=20, value=5, key="max_iterations"),
                    control=st.checkbox("Manual prompts between iterations", key="control_prompts"),
                    log_verbosity=st.select_slider(
                        "Log verbosity",
                        options=["BRIEF", "INFO", "DETAIL"],
                        value="BRIEF",
                        key="log_verbosity",
                    )
                )
            
            # 历史管理
            task_count = sum(1 for msg in st.session_state.messages if msg.type == "task_log")
            if task_count > 0:
                st.markdown(f"### 📋 Task History ({task_count} tasks)")
                if st.button("🗑️ Clear All History"):
                    st.session_state.messages = [Message.text("Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start.")]
                    st.rerun()
                    
        return config
    
    @staticmethod
    def render_single_message(msg):
        """Render a single message as a fragment to isolate interactions"""
        if msg.type == "text":
            st.write(msg.content["text"])
        elif msg.type == "user_summary":
            st.markdown(msg.content["summary"])
        elif msg.type == "command":
            st.code(msg.content["command"], language="bash")
        elif msg.type == "task_log":
            content = msg.content
            st.caption(f"ID: {content['run_id'][:8]}... | Completed: {content['timestamp']}")
            render_task_logs(
                content["phase_states"],
                content["max_iter"],
                show_progress=False
            )
        elif msg.type == "task_results":
            content = msg.content
            if "output_dir" in content and content["output_dir"]:
                from autogluon.assistant.webui.result_manager import ResultManager
                manager = ResultManager(content["output_dir"], content["run_id"])
                manager.render()
    
    @staticmethod
    def render_messages():
        """渲染消息历史"""
        for msg in st.session_state.messages:
            with st.chat_message(msg.role):
                UI.render_single_message(msg)
    
    @staticmethod
    def format_user_summary(files: List[str], config: TaskConfig, prompt: str, config_file: str) -> str:
        """格式化用户输入摘要"""
        parts = [
            "📂 **Uploaded files:**",
            "\n".join(f"- {f}" for f in files) if files else "- (none)",
            "\n⚙️ **Settings:**\n",
            f"- Config file: {config_file}",
            f"- Max iterations: {config.max_iter}",
            f"- Manual prompts: {config.control}",
            f"- Log verbosity: {config.log_verbosity}",
            "\n✏️ **Initial prompt:**\n",
            f"> {prompt or '(none)'}"
        ]
        return "\n".join(parts)


# ==================== Task Manager ====================
class TaskManager:
    """任务管理器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
    
    def handle_submission(self, submission):
        """处理用户提交"""
        # If waiting for input, handle it as iteration input
        if st.session_state.waiting_for_input:
            self.handle_iteration_input(submission)
            return
        
        # When accept_file="multiple", submission has .files and .text attributes
        files = submission.files or []
        user_text = submission.text.strip() if submission.text else ""
        
        if not files:
            SessionState.add_message(Message.text("⚠️ No data files provided. Please drag and drop your data files or ZIP."))
            st.rerun()
            return
        
        # 处理文件
        data_folder = handle_uploaded_files(files)
        st.session_state.data_src = data_folder
        
        # 保存配置文件
        config_path = self._save_config(data_folder)
        config_name = self.config.uploaded_config.name if self.config.uploaded_config else "default.yaml"
        
        # 添加用户摘要
        summary = UI.format_user_summary(
            [f.name for f in files],
            self.config,
            user_text,
            config_name
        )
        SessionState.add_message(Message.user_summary(summary, input_dir=data_folder))
        
        # 启动任务
        self._start_task(data_folder, config_path, user_text)
    
    def handle_iteration_input(self, submission):
        """处理迭代输入"""
        # 立即清空 iteration 信息显示，防止闪动
        SessionState.clear_iteration_info()
        
        # When accept_file=False, submission is just a string
        user_input = submission.strip() if submission else ""
        
        # Send input to backend
        if BackendAPI.send_user_input(st.session_state.run_id, user_input):
            # Clear waiting state
            SessionState.set_waiting_for_input(False)
            
            # Force clear the log processor state
            run_id = st.session_state.run_id
            processor_key = f"log_processor_{run_id}"
            if processor_key in st.session_state:
                processor = st.session_state[processor_key]
                processor.waiting_for_input = False
                processor.input_prompt = None
            
            # Small delay to ensure backend processes the input
            time.sleep(0.1)
            
            # Force fetch new logs
            new_logs = BackendAPI.fetch_logs(st.session_state.run_id)
            if new_logs:
                st.session_state.current_task_logs.extend(new_logs)
        else:
            SessionState.add_message(Message.text("❌ Failed to send input to the process."))
        
        st.rerun()
    
    def handle_cancel_request(self):
        """处理取消请求"""
        run_id = st.session_state.run_id
        if not run_id:
            return
        
        # 清空 iteration 信息
        SessionState.clear_iteration_info()
        
        # 显示用户的取消命令
        SessionState.add_message(Message.text("cancel", role="user"))
        
        # 尝试取消任务
        if BackendAPI.cancel_task(run_id):
            SessionState.add_message(Message.text(f"🛑 Task {run_id[:8]}... has been cancelled."))
            # 保存当前已有的日志
            if st.session_state.current_task_logs:
                processed = process_logs(
                    st.session_state.current_task_logs,
                    st.session_state.running_config.max_iter
                )
                
                # Extract output directory if available
                output_dir = self._extract_output_dir(processed["phase_states"])
                
                SessionState.add_message(
                    Message.task_log(
                        st.session_state.run_id,
                        processed["phase_states"],
                        st.session_state.running_config.max_iter,
                        output_dir,
                        st.session_state.current_input_dir
                    )
                )
                
                # Add task results message if output directory found
                if output_dir:
                    SessionState.add_message(
                        Message.task_results(st.session_state.run_id, output_dir)
                    )
                    
            SessionState.finish_task()
        else:
            SessionState.add_message(Message.text("❌ Failed to cancel the task."))
        
        st.rerun()
    
    def handle_task_deletion(self):
        """处理任务删除请求"""
        # Check for deletion flags
        keys_to_check = [k for k in st.session_state if k.startswith("delete_task_")]
        
        for key in keys_to_check:
            if st.session_state.get(key):
                run_id = key.replace("delete_task_", "")
                
                # Find the task messages to get directories
                output_dir = None
                input_dir = None
                
                for msg in st.session_state.messages:
                    if msg.type == "task_log" and msg.content.get("run_id") == run_id:
                        output_dir = msg.content.get("output_dir")
                        input_dir = msg.content.get("input_dir")
                        break
                
                # Delete directories
                success = True
                error_msg = ""
                
                try:
                    # Delete output directory
                    if output_dir and Path(output_dir).exists():
                        shutil.rmtree(output_dir)
                    
                    # Delete input directory
                    if input_dir and Path(input_dir).exists():
                        shutil.rmtree(input_dir)
                except Exception as e:
                    success = False
                    error_msg = str(e)
                
                # Remove from message history
                SessionState.delete_task_from_history(run_id)
                
                # Clear the deletion flag
                del st.session_state[key]
                
                # Show result message
                if success:
                    st.success(f"Task {run_id[:8]}... and all associated data have been deleted.")
                else:
                    st.error(f"Error deleting task data: {error_msg}")
                
                # Force a complete rerun
                st.rerun()
    
    def render_running_task(self):
        """Render the currently running task (without iteration info)"""
        if not st.session_state.task_running or not st.session_state.run_id:
            return
        
        run_id = st.session_state.run_id
        config = st.session_state.running_config
        
        if not config:
            st.error("Running configuration not found!")
            return
        
        # Only fetch new logs if not waiting for input
        if not st.session_state.waiting_for_input:
            new_logs = BackendAPI.fetch_logs(run_id)
            if new_logs:
                st.session_state.current_task_logs.extend(new_logs)
        
        # Get status
        status = BackendAPI.check_status(run_id)
        
        # Display running task
        with st.chat_message("assistant"):
            st.markdown(f"### Current Task")
            
            # Show different caption based on state
            if st.session_state.waiting_for_input:
                st.caption(f"ID: {run_id[:8]}... | Waiting for your input below...")
            else:
                st.caption(f"ID: {run_id[:8]}... | Type 'cancel' to stop the task")
            
            # Process logs (but don't show iteration info here)
            waiting_for_input, input_prompt = messages(st.session_state.current_task_logs, config.max_iter)
            
            # Update session state if waiting for input
            if waiting_for_input and not st.session_state.waiting_for_input:
                iteration = self._extract_current_iteration()
                SessionState.set_waiting_for_input(True, input_prompt, iteration)
                # 保存 iteration 信息内容
                st.session_state.iteration_info_content = self._extract_iteration_info()
                st.rerun()
        
        # Check if finished
        if status.get("finished", False):
            self._complete_task()
            st.rerun()
    
    def render_iteration_info(self, container):
        """在指定容器中渲染 iteration 信息"""
        if not st.session_state.show_iteration_info or st.session_state.clearing_iteration_info:
            container.empty()
            return
        
        with container.container():
            # 显示提示信息
            if st.session_state.input_prompt:
                st.info(f"💬 {st.session_state.input_prompt}")
            
            # 显示上一个 iteration 的信息
            if st.session_state.iteration_info_content:
                st.markdown("### Previous Iteration Results")
                
                # 获取前一个 iteration 的日志
                prev_logs = self._get_previous_iteration_logs()
                if prev_logs:
                    with st.expander("View previous iteration output", expanded=True):
                        for log in prev_logs:
                            st.write(log)
                
                st.markdown("---")
    
    def monitor_running_task(self):
        """监控运行中的任务"""
        if not st.session_state.task_running:
            return
        
        # 创建 iteration 信息容器
        iteration_container = st.empty()
        
        # 渲染 iteration 信息（如果需要）
        self.render_iteration_info(iteration_container)
        
        # Use fragment for task display
        @st.fragment(run_every=0.5 if not st.session_state.waiting_for_input else None)
        def update_task_display():
            self.render_running_task()
        
        update_task_display()
    
    def _extract_current_iteration(self) -> int:
        """Extract current iteration number from logs"""
        for entry in reversed(st.session_state.current_task_logs[-20:]):
            text = entry.get("text", "")
            if "Starting iteration" in text:
                try:
                    import re
                    match = re.search(r'Starting iteration (\d+)!', text)
                    if match:
                        return int(match.group(1))
                except:
                    pass
        return 1
    
    def _extract_iteration_info(self) -> Optional[str]:
        """提取需要显示的 iteration 信息"""
        # 这里可以根据需要定制要显示的信息
        iteration = st.session_state.current_iteration
        if iteration > 1:
            return f"Iteration {iteration - 1} completed"
        return None
    
    def _get_previous_iteration_logs(self) -> List[str]:
        """获取前一个 iteration 的关键日志"""
        if not st.session_state.current_task_logs:
            return []
        
        # 提取前一个 iteration 的执行结果
        relevant_logs = []
        current_iter = st.session_state.current_iteration
        
        if current_iter > 1:
            # 查找前一个 iteration 的日志
            in_prev_iter = False
            for entry in st.session_state.current_task_logs:
                text = entry.get("text", "")
                
                # 检测前一个 iteration 的开始
                if f"Starting iteration {current_iter - 1}!" in text:
                    in_prev_iter = True
                    continue
                
                # 检测当前 iteration 的开始（结束收集）
                if f"Starting iteration {current_iter}!" in text:
                    break
                
                # 收集相关日志
                if in_prev_iter and entry.get("level") in ["BRIEF", "INFO", "ERROR"]:
                    # 过滤掉一些不必要的日志
                    if not any(skip in text for skip in ["Previous iteration files", "Enter your inputs"]):
                        relevant_logs.append(text)
        
        # 返回最后 10 条相关日志
        return relevant_logs[-10:] if relevant_logs else []
    
    def _save_config(self, data_folder: str) -> str:
        """保存配置文件"""
        if self.config.uploaded_config:
            config_path = Path(data_folder) / self.config.uploaded_config.name
            with open(config_path, "wb") as f:
                f.write(self.config.uploaded_config.getbuffer())
            return str(config_path)
        return str(DEFAULT_CONFIG_PATH)
    
    def _start_task(self, data_folder: str, config_path: str, user_prompt: str):
        """启动任务"""
        # 先生成 run_id
        run_id = BackendAPI.start_task(data_folder, config_path, user_prompt, self.config)
        
        # 构建命令
        cmd_parts = [
            "mlzero",
            "-i", data_folder,
            "-n", str(self.config.max_iter),
            "-v", VERBOSITY_MAP[self.config.log_verbosity],
            "-c", config_path,
        ]
        
        if user_prompt:
            cmd_parts.extend(["-u", user_prompt])
        if self.config.control:
            cmd_parts.append("--need-user-input")
        
        # 显示命令
        command_str = f"[{datetime.now().strftime('%H:%M:%S')}] Running AutoMLAgent: {' '.join(cmd_parts)}"
        SessionState.add_message(Message.command(command_str))
        
        # 启动任务
        SessionState.start_task(run_id, self.config, data_folder)
        st.rerun()
    
    def _extract_output_dir(self, phase_states: Dict) -> Optional[str]:
        """Extract output directory from phase states"""
        output_phase = phase_states.get("Output", {})
        logs = output_phase.get("logs", [])
        
        for log in reversed(logs):
            import re
            match = re.search(r'output saved in\s+([^\s]+)', log)
            if match:
                output_dir = match.group(1).strip()
                output_dir = output_dir.rstrip('.,;:')
                return output_dir
        return None
    
    def _complete_task(self):
        """完成任务"""
        # 清空 iteration 信息
        SessionState.clear_iteration_info()
        
        # 保存任务日志
        if st.session_state.current_task_logs:
            processed = process_logs(
                st.session_state.current_task_logs,
                st.session_state.running_config.max_iter
            )
            
            # Extract output directory
            output_dir = self._extract_output_dir(processed["phase_states"])
            
            SessionState.add_message(
                Message.task_log(
                    st.session_state.run_id,
                    processed["phase_states"],
                    st.session_state.running_config.max_iter,
                    output_dir,
                    st.session_state.current_input_dir
                )
            )
            
            # Add success message
            SessionState.add_message(Message.text(SUCCESS_MESSAGE))
            
            # Add task results message if output directory found
            if output_dir:
                SessionState.add_message(
                    Message.task_results(st.session_state.run_id, output_dir)
                )
        
        SessionState.finish_task()


# ==================== Main App ====================
class AutoMLAgentApp:
    """主应用"""
    
    def __init__(self):
        UI.setup_page()
        SessionState.init()
        self.config = UI.render_sidebar()
        self.task_manager = TaskManager(self.config)
    
    def run(self):
        """运行应用"""
        # Check for task deletion requests first
        self.task_manager.handle_task_deletion()
        
        # 渲染历史消息
        UI.render_messages()
        
        # 监控运行中的任务（包括 iteration 信息显示）
        self.task_manager.monitor_running_task()
        
        # Determine chat input configuration based on state
        if st.session_state.waiting_for_input:
            # When waiting for iteration input
            placeholder = "Enter your input for this iteration (press Enter to skip)"
            accept_file = False
        elif st.session_state.task_running:
            # When task is running but not waiting for input
            placeholder = "Type 'cancel' to stop the current task"
            accept_file = False
        else:
            # Normal state - ready to accept new tasks
            placeholder = "Type optional prompt, or drag & drop your data files/ZIP here"
            accept_file = "multiple"
        
        # 处理用户输入
        submission = st.chat_input(
            placeholder=placeholder,
            accept_file=accept_file,
            key="u_input",
            max_chars=10000,
        )
        
        if submission:
            # 如果正在等待输入
            if st.session_state.waiting_for_input:
                self.task_manager.handle_iteration_input(submission)
            # 如果任务正在运行
            elif st.session_state.task_running:
                # 检查是否是取消命令
                if submission and submission.strip().lower() == "cancel":
                    self.task_manager.handle_cancel_request()
                else:
                    # 显示提示信息
                    SessionState.add_message(
                        Message.text(
                            "⚠️ A task is currently running. Type 'cancel' to stop it, or wait for it to complete.",
                            role="user"
                        )
                    )
                    st.rerun()
            else:
                # 没有任务运行，正常处理提交
                self.task_manager.handle_submission(submission)


def main():
    """入口点"""
    app = AutoMLAgentApp()
    app.run()


if __name__ == "__main__":
    main()