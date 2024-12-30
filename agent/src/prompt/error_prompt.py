def generate_error_prompt(error_message, max_error_message_length):
    if not error_message:
        return ""

    # TODO: Simplify the error message
    error_prompt = f"While running previous program, following error occurred:\n\n{error_message}\n\n"

    return error_prompt
