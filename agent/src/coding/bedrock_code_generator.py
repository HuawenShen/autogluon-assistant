import boto3
import json
import os
import argparse

from .utils import read_prompt, extract_python_script, write_code_script, save_script


def call_claude_bedrock(prompt, model_id):
    # Create a Bedrock client
    # The SDK will automatically use the instance profile credentials
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )

    # Prepare the request body for the Messages API
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "top_p": 1,
        }
    )

    # Call the Bedrock API
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,  # Use the appropriate model ID
        accept="application/json",
        contentType="application/json",
    )

    # Parse and return the response
    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


[]


def use_bedrock_to_generate(prompt, model_id):
    # Call Claude in Bedrock
    response = call_claude_bedrock(prompt, model_id)

    # Extract the Python script from the response
    code_script = extract_python_script(response)

    ret = {
        "code_script": code_script,
    }

    return ret


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract Python script from Claude response"
    )
    parser.add_argument(
        "-p", "--prompt_file", required=True, help="Path to the prompt file"
    )
    parser.add_argument(
        "-c", "--output_code_file", required=True, help="Path to the output code file"
    )
    parser.add_argument(
        "-m",
        "--model_id",
        required=False,
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help="Claude model ID to use",
    )
    args = parser.parse_args()

    # Read the prompt from the file
    prompt = read_prompt(args.prompt_file)

    script = use_bedrock_to_generate(prompt, args.model_id)

    write_code_script(script, args.output_code_file)


if __name__ == "__main__":
    main()
