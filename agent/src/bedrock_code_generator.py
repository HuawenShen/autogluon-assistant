import boto3
import json
import re
import os
import argparse

def read_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def extract_python_script(response):
    # Look for Python code blocks in the response
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    else:
        return None

def save_script(script, output_file):
    with open(output_file, 'w') as file:
        file.write(script)

def call_claude_bedrock(access_key, secret_access_key, prompt, model_id):
    # Create a Bedrock client
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_access_key,
        region_name='us-east-1'  # Adjust the region as needed
    )

    # Prepare the request body for the Messages API
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "top_p": 1
    })

    # Call the Bedrock API
    response = bedrock.invoke_model(
        body=body,
        modelId='anthropic.claude-3-haiku-20240307-v1:0',  # Use the appropriate model ID
        accept='application/json',
        contentType='application/json'
    )

    # Parse and return the response
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract Python script from Claude response')
    parser.add_argument('-p', '--prompt_file', required=True, help='Path to the prompt file')
    parser.add_argument('-c', '--output_code_file', required=True, help='Path to the output code file')
    parser.add_argument('-m', '--model_id', required=False, default='anthropic.claude-3-haiku-20240307-v1:0', help='Claude model ID to use')
    args = parser.parse_args()

    # Get AWS credentials from environment variables
    access_key = os.environ.get('BEDROCK_ACCESS_KEY')
    secret_access_key = os.environ.get('BEDROCK_SECRET_ACCESS_KEY')

    if not access_key or not secret_access_key:
        raise ValueError("AWS credentials not found in environment variables. "
                         "Please set BEDROCK_ACCESS_KEY and BEDROCK_SECRET_ACCESS_KEY.")

    # Read the prompt from the file
    prompt = read_prompt(args.prompt_file)

    # Call Claude in Bedrock
    response = call_claude_bedrock(access_key, secret_access_key, prompt, args.model_id)

    # Extract the Python script from the response
    script = extract_python_script(response)

    if script:
        # Save the extracted script to the output file
        save_script(script, args.output_code_file)
        print(f"Python script extracted and saved to {args.output_code_file}")
    else:
        print("No Python script found in the response.")

if __name__ == "__main__":
    main()