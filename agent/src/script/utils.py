import re


def write_code_script(script, output_code_file):
    if script:
        # Save the extracted script to the output file
        save_script(script, output_code_file)
        print(f"Python script extracted and saved to {output_code_file}")
    else:
        print("No Python script found in the response.")


def save_script(script, output_file):
    with open(output_file, "w") as file:
        file.write(script)


def extract_python_script(response):
    # Look for Python code blocks in the response
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return matches[0].strip()
    else:
        return None


def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()
