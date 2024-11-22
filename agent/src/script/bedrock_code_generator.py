import boto3
import json
import os
import argparse

from .utils import read_prompt, extract_python_script, write_code_script, save_script


def call_claude_bedrock(access_key, secret_access_key, prompt, model_id):
    '''
    # Create a session using your default credentials
    session = boto3.Session()

    # Create an STS client
    sts_client = session.client('sts')

    # Assume the IAM role
    assumed_role_object = sts_client.assume_role(
        RoleArn="arn:aws:iam::097403188315:role/Bedrock_Access",
        RoleSessionName="AssumeRoleSession"
    )

    # Get the temporary credentials
    credentials = assumed_role_object['Credentials']

    # Use these credentials to create a Bedrock client
    bedrock = boto3.client(
        'bedrock',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )
    '''

    # Create a Bedrock client
    # The SDK will automatically use the instance profile credentials
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )

    # Create a Bedrock client
    #bedrock = boto3.client(
    #    service_name="bedrock-runtime",
    #    aws_access_key_id=access_key,
    #    aws_secret_access_key=secret_access_key,
    #    region_name="us-east-1",  # Adjust the region as needed
    #)

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
    # Get AWS credentials from environment variables
    access_key = os.environ.get("BEDROCK_ACCESS_KEY")
    secret_access_key = os.environ.get("BEDROCK_SECRET_ACCESS_KEY")

    if not access_key or not secret_access_key:
        raise ValueError(
            "AWS credentials not found in environment variables. "
            "Please set BEDROCK_ACCESS_KEY and BEDROCK_SECRET_ACCESS_KEY."
        )

    # Call Claude in Bedrock
    response = call_claude_bedrock(access_key, secret_access_key, prompt, model_id)

    # Extract the Python script from the response
    script = extract_python_script(response)

    return script


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
