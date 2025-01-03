from typing import List

def generate_code_history_prompt(code_scripts: List[str], max_code_history_length: int) -> str:
    if not code_scripts:
        return ""
    
    code_history_prompt = "Previous Code:\n\n"
    for code_script in code_scripts:
        code_history_prompt += f"{code_script}\n\n"
        
    # Truncate the prompt if it exceeds max length
    return code_history_prompt[:max_code_history_length]
