from .agrag_code_generator import use_agrag_to_generate
from .bedrock_code_generator import use_bedrock_to_generate
from .utils import write_code_script, write_retrieved_context


def generate_code(prompt, model_id, backend="bedrock", tutorial_link=None):
    if backend == "bedrock":
        return use_bedrock_to_generate(prompt, model_id, mode="python")
    elif backend == "agrag":
        return use_agrag_to_generate(prompt, model_id, tutorial_link)


def generate_script(prompt, model_id, backend="bedrock", tutorial_link=None):
    if backend == "bedrock":
        return use_bedrock_to_generate(prompt, model_id, mode="bash")
    elif backend == "agrag":
        raise NotImplementedError
