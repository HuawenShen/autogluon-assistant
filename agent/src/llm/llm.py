import logging
import os
from typing import Any, Dict, List

import boto3
from autogluon.assistant.constants import WHITE_LIST_LLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseAssistantChat(BaseModel):
    """Base class for assistant chat models with conversation support."""

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_tokens_: int = Field(default=0)
    output_tokens_: int = Field(default=0)
    conversation_: ConversationChain = None

    def initialize_conversation(
        self,
        llm,
        system_prompt="You are a technical assistant that excels at working on data science tasks.",
    ):
        memory = ConversationBufferMemory()
        self.conversation_ = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=llm.verbose,
            prompt=ChatPromptTemplate.from_messages(
                [SystemMessage(content=system_prompt), HumanMessage(content="{input}")]
            ),
        )

    def describe(self) -> Dict[str, Any]:
        """Get model description and conversation history."""
        return {
            "history": self.history_,
            "prompt_tokens": self.input_tokens_,
            "completion_tokens": self.output_tokens_,
            "conversation": self.conversation_.memory.load_memory_variables({}),
        }

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def __call__(self, message: str) -> str:
        """Send a message and get response in conversation context."""
        response = self.conversation_.run(message)
        if hasattr(response, "usage_metadata"):
            self.input_tokens_ += response.usage_metadata.get("input_tokens", 0)
            self.output_tokens_ += response.usage_metadata.get("output_tokens", 0)

        # Record in history
        self.history_.append(
            {
                "input": message,
                "output": response,
                "prompt_tokens": self.input_tokens_,
                "completion_tokens": self.output_tokens_,
            }
        )

        return response


class AssistantChatOpenAI(ChatOpenAI, BaseAssistantChat):
    """OpenAI chat model with conversation support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {
            **base_desc,
            "model": self.model_name,
            "proxy": self.openai_proxy,
        }


class AssistantChatBedrock(ChatBedrock, BaseAssistantChat):
    """Bedrock chat model with conversation support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {
            **base_desc,
            "model": self.model_id,
        }


class ChatLLMFactory:
    @staticmethod
    def get_openai_models() -> List[str]:
        try:
            client = OpenAI()
            models = client.models.list()
            return [
                model.id
                for model in models
                if model.id.startswith(("gpt-3.5", "gpt-4"))
            ]
        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
            return []

    @staticmethod
    def get_bedrock_models() -> List[str]:
        try:
            bedrock = boto3.client("bedrock", region_name="us-west-2")
            response = bedrock.list_foundation_models()
            return [model["modelId"] for model in response["modelSummaries"]]
        except Exception as e:
            print(f"Error fetching Bedrock models: {e}")
            return []

    @classmethod
    def get_chat_model(cls, config: DictConfig) -> BaseAssistantChat:
        """Get a configured chat model instance."""
        provider = config.provider
        model = config.model

        # Validate provider
        valid_providers = ["openai", "bedrock"]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {provider}. Must be one of {valid_providers}"
            )

        # Validate model
        valid_models = (
            cls.get_openai_models()
            if provider == "openai"
            else cls.get_bedrock_models()
        )
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model} for provider {provider}")

        if model not in WHITE_LIST_LLM:
            logger.warning(f"Model {model} is not on the white list: {WHITE_LIST_LLM}")

        # Create appropriate model instance
        if provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise Exception("OpenAI API key not found in environment")

            logger.info(f"Using OpenAI model: {model}")
            return AssistantChatOpenAI(
                model_name=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                verbose=config.verbose,
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_base=config.proxy_url,
            )
        else:  # bedrock
            logger.info(f"Using Bedrock model: {model}")
            return AssistantChatBedrock(
                model_id=model,
                model_kwargs={
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
                region_name="us-west-2",
                verbose=config.verbose,
            )
