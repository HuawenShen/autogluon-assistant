import re
from rich import print


def write_code_script(script, output_code_file):
    if script:
        # Save the extracted script to the output file
        save_script(script, output_code_file)
        print(f"[bold green]Python script extracted and saved to[/bold green] {output_code_file}")
    else:
        print("[bold yellow]No Python script found in the response.[/bold yellow]")


def write_retrieved_context(retrieved_context, output_context_path):
    if retrieved_context:
        # Save the extracted script to the output file
        save_retrieved_context(retrieved_context, output_context_path)
        print(f"[bold green]Context retrieved and saved to[/bold green] {output_context_path}")
    else:
        print("[bold yellow]No retrieved context.[/bold yellow]")


def save_retrieved_context(retrieved_context, output_file):
    with open(output_file, "w") as file:
        for context in retrieved_context:
            file.write(context)


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
        print(f"[bold yellow]No python script found in reponse, return the full response instead:[/bold yellow] {response}")
        return response


def extract_bash_script(response):
    # Look for Bash code blocks in the response
    pattern = r"```bash\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        print(f"[bold yellow]No bash script found in reponse, return the full response instead:[/bold yellow] {response}")
        return response


def extract_script(response, language):
    if language == "python":
        return extract_python_script(response)
    elif language == "bash":
        return extract_bash_script(response)
    else:
        raise ValueError(f"Unsupported mode: {language}")


def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

