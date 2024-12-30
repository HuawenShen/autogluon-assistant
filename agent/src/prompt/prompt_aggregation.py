import logger
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

from .data_prompt import generate_data_prompt
from .error_prompt import generate_error_prompt
from .task_prompt import generate_task_prompt
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
    max_user_inputs_length: int = 9999
    max_error_message_length: int = 9999 
    max_tutorial_length: int = 9999
    llm: LLMConfig = LLMConfig()

class PromptGenerator:
    def __init__(
        self, 
        input_data_folder: str,
        tutorials_folder: str,
        output_folder: str,
        config_path: str
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
        for path, name in [(input_data_folder, "input_data_folder"), 
                          (tutorials_folder, "tutorials_folder")]:
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

        self.user_prompts: List[str] = []
        self.error_prompts: List[str] = []
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
            llm_config=OmegaConf.to_container(self.config.llm, resolve=True),
        )
        
        # TODO: use LLM to select a task prompt from tabular/automm/timeseries

        return {
            "task_prompt": task_prompt,
            "data_prompt": data_prompt,
        }

    @property
    def user_prompt(self) -> str:
        if self.time_step >= 0:
            return self.user_prompts[self.time_step]
        else:
            logger.warning(
                "No user prompt because the prompt generator is not stepped yet."
            )
            return ""

    @property
    def error_prompt(self) -> str:
        if self.time_step >= 0:
            return self.error_prompts[self.time_step]
        else:
            logger.warning(
                "No error prompt because the prompt generator is not stepped yet."
            )
            return ""

    @property
    def tutorial_prompt(self) -> str:
        if self.time_step >= 0:
            return self.tutorial_prompts[self.time_step]
        else:
            logger.warning(
                "No tutorial prompt because the prompt generator is not stepped yet."
            )
            return ""

    def step(self, user_inputs=None, error_message=None):
        """Step the prompt generator forward.
        
        Args:
            user_inputs: Optional user inputs to generate user prompt
            error_message: Optional error message to generate error prompt
        """
        user_prompt = generate_user_prompt(
            user_inputs=user_inputs,
            max_user_inputs_length=self.config.max_user_inputs_length,
        )
        error_prompt = generate_error_prompt(
            error_message=error_message,
            max_error_message_length=self.config.max_error_message_length,
        )
        tutorial_prompt = generate_tutorial_prompt(
            task_prompt=self.task_prompt,
            data_prompt=self.data_prompt,
            user_prompt=user_prompt,
            error_prompt=error_prompt,
            tutorial_folder=self.tutorials_folder,
            llm_config=OmegaConf.to_container(self.config.llm, resolve=True),
            output_folder=self.output_folder,
            max_num_tutorials=self.config.max_num_tutorials,
            max_tutorial_length=self.config.max_tutorial_length,
        )

        self.user_prompts.append(user_prompt)
        self.error_prompts.append(error_prompt)
        self.tutorial_prompts.append(tutorial_prompt)

        self.time_step += 1

    def get_iterative_prompt(self) -> str:
        """Get the complete iterative prompt.
        
        Returns:
            str: The complete prompt combining task, data, user, error and tutorial prompts
        """
        assert (
            self.time_step >= 0
        ), "run PromptGenerator.step(user_inputs, error_message) before get the prompt"
        
        prompt_parts = []
        
        if self.time_step == 0:
            prompt_parts.extend([self.task_prompt, self.data_prompt])

        if self.user_prompt:
            prompt_parts.append(self.user_prompt)

        if self.error_prompt:
            prompt_parts.append(self.error_prompt)

        if self.tutorial_prompt:
            prompt_parts.append(self.tutorial_prompt)

        return "\n\n".join(prompt_parts)
