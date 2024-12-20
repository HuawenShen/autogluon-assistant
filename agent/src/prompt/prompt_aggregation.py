import logger

from .data_prompt_generator import generate_data_prompt
from .task_prompt_generator import generate_task_prompt
from .tutorial_prompt_generator import generate_tutorial_prompt
from .user_prompt_generator import generate_user_prompt
from .error_prompt_generator import generate_error_prompt


class PromptGenerator:

    def __init__(
        self, input_data_folder, tutorials_folder, output_folder, max_chars_per_file=100
    ):
        self.input_data_folder = input_data_folder
        self.tutorials_folder = tutorials_folder
        self.output_folder = output_folder
        self.max_chars_per_file = max_chars_per_file

        initial_prompts = self.generate_initial_prompts()
        self.task_prompt = initial_prompts["task_prompt"]
        self.data_prompt = initial_prompts["data_prompt"]
        self.tutorials_sub_folder = initial_prompts["tutorials_sub_folder"]

        self.user_prompts = []
        self.error_prompts = []
        self.tutorial_prompts = []

        self.time_step = -1

    def generate_initial_prompts(
        self,
    ):
        data_prompt = generate_data_prompt(
            input_data_folder=self.input_data_folder,
            max_chars_per_file=self.max_chars_per_file,
        )
        task_prompt, tutorials_sub_folder = generate_task_prompt(
            data_prompt=data_prompt, output_folder=self.output_folder
        )  # use LLM to select a task prompt from tabular/automm/timeseries

        return {
            "task_prompt": task_prompt,
            "data_prompt": data_prompt,
            "tutorials_sub_folder": tutorials_sub_folder,
        }

    @property
    def user_prompt(
        self,
    ):
        if self.time_step >= 0:
            return self.user_prompts[self.time_step]
        else:
            logger.warning(
                "No user prompt because the prompt generator is not stepped yet."
            )
            return ""

    @property
    def error_prompt(
        self,
    ):
        if self.time_step >= 0:
            return self.error_prompts[self.time_step]
        else:
            logger.warning(
                "No error prompt because the prompt generator is not stepped yet."
            )
            return ""

    @property
    def tutorial_prompt(
        self,
    ):
        if self.time_step >= 0:
            return self.tutorial_prompts[self.time_step]
        else:
            logger.warning(
                "No tutorial prompt because the prompt generator is not stepped yet."
            )
            return ""

    def step(
        self,
        user_inputs=None,
        error_message=None,
        max_num_tutorials=3,
        max_user_inputs_length=9999,
        max_error_message_length=9999,
        max_tutorial_length=9999,
    ):
        user_prompt = generate_user_prompt(
            user_inputs=user_inputs,
            max_user_inputs_length=max_user_inputs_length,
        )
        error_prompt = generate_user_prompt(
            error_message=error_message,
            max_error_message_length=max_error_message_length,
        )
        tutorial_prompt = generate_tutorial_prompt(
            task_prompt=self.task_prompt,
            data_prompt=self.data_prompt,
            user_prompt=user_prompt,
            error_prompt=error_prompt,
            max_num_tutorials=max_num_tutorials,
            max_tutorial_length=max_tutorial_length,
        )

        self.user_prompts.append(user_prompt)
        self.error_prompts.append(error_prompt)
        self.tutorial_prompts.append(tutorial_prompt)

        self.time_step += 1

    def get_iterative_prompt(
        self,
    ):
        assert (
            self.time_step >= 0
        ), f"run PromptGenerator.step(user_inputs, error_message) before get the prompt"
        prompt = ""
        if self.time_step == 0:
            prompt += f"{self.task_prompt}\n\n{self.data_prompt}\n\n"

        if self.user_prompt:
            prompt += f"{self.user_prompt}\n\n"

        if self.error_prompt:
            prompt += f"{self.error_prompt}\n\n"

        if self.tutorial_prompt:
            prompt += f"{self.tutorial_prompt}\n\n"

        raise NotImplementedError
