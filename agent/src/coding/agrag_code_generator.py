import boto3
import json
import os
import argparse

from agrag.agrag import AutoGluonRAG
from agrag.modules.generator.utils import format_query

from .utils import read_prompt, extract_python_script, write_code_script, save_script


# TODO: Implement Query Generator
def generate_query(prompt):
    return prompt


def use_agrag_to_generate(prompt, model_id, tutorial_link):
    base_url = "/".join(tutorial_link.split("/")[:-1]) + "/"
    agrag = AutoGluonRAG(
        config_file="/media/deephome/AutoMLAgent/agent/src/configs/agrag_semantic_segmentation.yaml", # or path to config file
        web_urls=[tutorial_link],  # Use the provided tutorial_link
        base_urls=[base_url],      # Use the derived base_url
        parse_urls_recursive=True,
        #data_dir="s3://autogluon-rag-github-dev/autogluon_docs/"
    )
    agrag.initialize_rag_pipeline()

    retrieved_context = ""
    if agrag.retriever_module.top_k > 0:
        retrieved_context = agrag.retrieve_context_for_query(prompt)

    # TODO: log retrieved_context

    query = generate_query(prompt)

    prompt_prefix = agrag.args.generator_query_prefix
    if prompt_prefix:
        query = f"{prompt_prefix}\n{query}"
    formatted_prompt = format_query(
        model_name=agrag.generator_module.model_name, query=prompt, context=retrieved_context
    )

    response = agrag.generator_module.generate_response(formatted_prompt)

    # Extract the Python script from the response
    code_script = extract_python_script(response)

    ret = {
        "code_script": code_script,
        "retrieved_context": retrieved_context,
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

    script = use_agrag_to_generate(prompt, args.model_id)

    write_code_script(script, args.output_code_file)


if __name__ == "__main__":
    main()
