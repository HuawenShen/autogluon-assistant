from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import logging

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)
from omegaconf import OmegaConf

from .data_prompt import generate_data_prompt
from .error_prompt import generate_error_prompt
from .code_history_prompt import generate_code_history_prompt
from .task_prompt import generate_task_prompt
from .execution_prompt import generate_execution_prompt
from .tutorial_prompt import generate_tutorial_prompt
from .user_prompt import generate_user_prompt


@dataclass
class LLMConfig:
    provider: str = "bedrock"
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    max_tokens: int = 512
    proxy_url: Optional[str] = None
    temperature: float = 0
    verbose: bool = True


@dataclass
class PromptGeneratorConfig:
    max_chars_per_file: int = 100
    max_num_tutorials: int = 3
    max_user_input_length: int = 9999
    max_error_message_length: int = 9999
    max_tutorial_length: int = 9999
    llm: LLMConfig = LLMConfig()
    create_venv: bool = False


class PromptGenerator:
    def __init__(
        self,
        input_data_folder: str,
        tutorials_folder: str,
        output_folder: str,
        config_path: str,
    ):
        """Initialize PromptGenerator with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            tutorials_folder: Path to tutorials directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        # Store required paths
        self.input_data_folder = input_data_folder
        self.tutorials_folder = tutorials_folder
        self.output_folder = output_folder

        # Validate paths
        for path, name in [
            (input_data_folder, "input_data_folder"),
            (tutorials_folder, "tutorials_folder"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Load config from YAML and merge with default
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        default_config = OmegaConf.structured(PromptGeneratorConfig)
        yaml_config = OmegaConf.load(config_path)
        self.config = OmegaConf.merge(default_config, yaml_config)

        # Initialize prompts
        initial_prompts = self.generate_initial_prompts()
        self.task_prompt = initial_prompts["task_prompt"]
        self.data_prompt = initial_prompts["data_prompt"]

        self.user_inputs: List[str] = []
        self.error_messages: List[str] = []
        self.python_codes: List[str] = []
        self.bash_scripts: List[str] = []
        self.tutorial_prompts: List[str] = []

        self.time_step = -1

    def generate_initial_prompts(self):
        data_prompt = generate_data_prompt(
            input_data_folder=self.input_data_folder,
            max_chars_per_file=self.config.max_chars_per_file,
        )

        task_prompt = generate_task_prompt(
            data_prompt=data_prompt,
            output_folder=self.output_folder,
            llm_config=self.config.llm,
        )

        # TODO: use LLM to select a task prompt from tabular/automm/timeseries

        return {
            "task_prompt": task_prompt,
            "data_prompt": data_prompt,
        }

    @property
    def user_input(self) -> str:
        assert self.time_step >= 0, "No user input because the prompt generator is not stepped yet."
        assert len(self.user_inputs) == self.time_step + 1, "user input is not updated yet"
        return self.user_inputs[self.time_step]

    @property
    def python_code(self) -> str:
        assert self.time_step >= 0, "No python code because the prompt generator is not stepped yet."
        assert len(self.python_codes) == self.time_step + 1, "python code is not updated yet"
        return self.python_codes[self.time_step]
    
    @property
    def previous_python_code(self) -> str:
        if self.time_step >= 1:
            return self.python_codes[self.time_step - 1]
        else:
            return ""

    @property
    def bash_script(self) -> str:
        assert self.time_step >= 0, "No bash script because the prompt generator is not stepped yet."
        assert len(self.bash_scripts) == self.time_step + 1, "bash script is not updated yet"
        return self.bash_scripts[self.time_step]
    
    @property
    def previous_bash_script(self) -> str:
        if self.time_step >= 1:
            return self.bash_scripts[self.time_step - 1]
        else:
            return ""

    @property
    def error_message(self) -> str:
        assert self.time_step >= 0, "No error message because the prompt generator is not stepped yet."
        assert len(self.error_messages) == self.time_step + 1, "error message is not updated yet"
        return self.error_messages[self.time_step]

    @property
    def previous_error_message(self) -> str:
        if self.time_step >= 1:
            return self.error_messages[self.time_step - 1]
        else:
            return ""

    @property
    def tutorial_prompt(self) -> str:
        assert self.time_step >= 0, "No tutorial prompt because the prompt generator is not stepped yet."
        assert len(self.tutorial_prompts) == self.time_step + 1, "tutorial prompt is not updated yet"
        return self.tutorial_prompts[self.time_step]

    def step(self, user_input=None):
        """Step the prompt generator forward.

        Args:
            user_inputs: Optional user inputs to generate user prompt
            error_message: Optional error message to generate error prompt
        """
        self.time_step += 1

        user_prompt = generate_user_prompt(
            user_input=user_input,
            max_user_input_length=self.config.max_user_input_length,
        )
        error_prompt = generate_error_prompt(
            error_message=self.previous_error_message,
            max_error_message_length=self.config.max_error_message_length,
        )
        tutorial_prompt = generate_tutorial_prompt(
            task_prompt=self.task_prompt,
            data_prompt=self.data_prompt,
            user_prompt=user_prompt,
            error_prompt=error_prompt,
            tutorial_folder=self.tutorials_folder,
            llm_config=self.config.llm,
            output_folder=self.output_folder,
            max_num_tutorials=self.config.max_num_tutorials,
            max_tutorial_length=self.config.max_tutorial_length,
        )

        assert len(self.user_inputs) == self.time_step
        self.user_inputs.append(user_input)

        assert len(self.tutorial_prompts) == self.time_step
        self.tutorial_prompts.append(tutorial_prompt)

    def get_coding_prompt(self) -> str:
        """Get the complete iterative prompt.

        Returns:
            str: The complete prompt combining task, data, user, error and tutorial prompts
        """
        assert (
            self.time_step >= 0
        ), "run PromptGenerator.step(user_input) before get the prompt"

        prompt_parts = []

        if self.time_step == 0:
            prompt_parts.extend([self.task_prompt, self.data_prompt])

        if self.user_input:
            user_prompt = generate_user_prompt(
                user_input=self.user_input,
                max_user_input_length=self.config.max_user_input_length,
            )
            prompt_parts.append(user_prompt)

        if self.previous_error_message:
            error_prompt = generate_error_prompt(
                error_message=self.previous_error_message,
                max_error_message_length=self.config.max_error_message_length,
            )
            prompt_parts.append(error_prompt)

        if self.tutorial_prompt:
            prompt_parts.append(self.tutorial_prompt)

        return "\n\n".join(prompt_parts)

    def get_execution_prompt(self, python_file_path) -> str:
        self.execution_prompt = generate_execution_prompt(
            output_folder=self.output_folder,
            python_file_path=python_file_path,
            create_venv=self.config.create_venv,
            previous_bash=self.previous_bash_script,
            previous_python=self.previous_python_code,
            current_python=self.python_code,
            error_message=self.previous_error_message,
        )
        return self.execution_prompt

    def update_python_code(self, python_code: str):
        """Update the current Python code."""
        assert len(self.user_inputs) == self.time_step
        self.python_codes.append(python_code)
        
    def update_bash_script(self, bash_script: str):
        """Update the current bash code."""
        assert len(self.user_inputs) == self.time_step
        self.bash_scripts.append(bash_script)
        
    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)
