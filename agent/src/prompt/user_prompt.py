from typing import Optional

def generate_user_prompt(
    user_inputs: Optional[str] = None,
    max_user_inputs_length: int = 9999,
) -> str:
    """Generate a formatted user prompt from user input.
    
    Args:
        user_inputs: User input string to include in the prompt.
                    If None, returns an empty string.
        max_user_inputs_length: Maximum allowed length for user input.
    
    Returns:
        str: Formatted user prompt with wrapped and truncated input.
    """
    if not user_inputs:
        return ""
        
    # Truncate if needed
    if len(user_inputs) > max_user_inputs_length:
        user_inputs = user_inputs[:max_user_inputs_length-3] + "..."

    # Create the prompt with section header
    prompt = f"USER INPUTS:\n{user_inputs.strip()}"
    
    return prompt
