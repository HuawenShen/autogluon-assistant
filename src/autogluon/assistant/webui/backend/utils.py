# src/autogluon/assistant/webui/backend/utils.py

import re
import subprocess
import threading

# 全局存储每个 run 的状态
_runs: dict = {}

def parse_log_line(line: str) -> dict:
    """
    解析一行原始日志，提取 level 和 text，两者均为字符串。
    支持可选前缀 [MM/DD/YY hh:mm:ss]，会自动丢弃。
    例子：
      "[05/29/25 22:56:33] INFO    Some message here module.py:123"
    或者：
      "BRIEF   Brief-level message"
    都能正确提取。
    """
    # 正则：可选时间戳、空格、级别（字母）、任意空白、正文
    m = re.match(
        r'''
        (?:\[\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*)?   # 可选时间戳 [MM/DD/YY hh:mm:ss]
        (?P<level>[A-Z]+)                                  # 日志级别，全大写字母
        \s+                                                # 分隔空格
        (?P<text>.*)                                       # 剩下的全部，作为正文
        ''',
        line,
        re.VERBOSE,
    )
    if not m:
        # 这里的问题，正则永远匹配不上
        return {"level": "INFO", "text": line}
    return {"level": m.group("level"), "text": m.group("text").strip()}

def start_run(run_id: str, cmd: list[str]):
    """
    启动子进程，并在后台线程中持续读取 stdout/stderr，
    将每一行 append 到 _runs[run_id]['logs']。
    """
    _runs[run_id] = {
        "process": None,
        "logs": [],
        "pointer": 0,
        "finished": False,
    }

    def _target():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _runs[run_id]["process"] = p
        for line in p.stdout:
            _runs[run_id]["logs"].append(line.rstrip("\n"))
        p.wait()
        _runs[run_id]["finished"] = True

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

def get_logs(run_id: str) -> list[str]:
    """
    返回自上次调用后新增的日志行列表。
    """
    info = _runs.get(run_id)
    if info is None:
        return []
    logs = info["logs"]
    ptr = info["pointer"]
    new = logs[ptr:]
    info["pointer"] = len(logs)
    return new

def get_status(run_id: str) -> dict:
    """
    返回任务是否完成。
    """
    info = _runs.get(run_id)
    if info is None:
        return {"finished": True, "error": "run_id not found"}
    return {"finished": info["finished"]}

def cancel_run(run_id: str):
    """
    终止对应 run 的子进程。
    """
    info = _runs.get(run_id)
    if info and info["process"]:
        info["process"].terminate()
        info["finished"] = True
