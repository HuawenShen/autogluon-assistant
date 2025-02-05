import logging
import os
from typing import Any, Dict, List, Optional
import uuid

import boto3
from autogluon.assistant.constants import WHITE_LIST_LLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class GlobalTokenTracker:
    """Singleton class to track token usage across all conversations."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalTokenTracker, cls).__new__(cls)
            cls._instance.total_input_tokens = 0
            cls._instance.total_output_tokens = 0
            cls._instance.conversations = {}  # Track per-conversation usage
        return cls._instance
    
    def add_tokens(self, conversation_id: str, input_tokens: int, output_tokens: int):
        """Add token counts for a specific conversation."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {"input_tokens": 0, "output_tokens": 0}
        
        self.conversations[conversation_id]["input_tokens"] += input_tokens
        self.conversations[conversation_id]["output_tokens"] += output_tokens
    
    def get_total_usage(self) -> Dict[str, int]:
        """Get total token usage across all conversations."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
    
    def get_conversation_usage(self, conversation_id: str) -> Dict[str, int]:
        """Get token usage for a specific conversation."""
        if conversation_id not in self.conversations:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        conv = self.conversations[conversation_id]
        return {
            "input_tokens": conv["input_tokens"],
            "output_tokens": conv["output_tokens"],
            "total_tokens": conv["input_tokens"] + conv["output_tokens"]
        }


class BaseAssistantChat(BaseModel):
    """Base class for assistant chat models with LangGraph support."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_tokens_: int = Field(default=0)
    output_tokens_: int = Field(default=0)
    graph: Optional[Any] = Field(default=None, exclude=True)
    app: Optional[Any] = Field(default=None, exclude=True)
    memory: Optional[Any] = Field(default=None, exclude=True)
    token_tracker: GlobalTokenTracker = Field(default_factory=GlobalTokenTracker)
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def initialize_conversation(
        self,
        llm: Any,
        system_prompt: str = "You are a technical assistant that excels at working on data science tasks.",
    ) -> None:
        """Initialize conversation using LangGraph."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        graph = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            prompt_messages = prompt_template.invoke(state)
            response = llm.invoke(prompt_messages)
            return {"messages": [response]}

        graph.add_edge(START, "model")
        graph.add_node("model", call_model)

        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)

        self.graph = graph
        self.app = app
        self.memory = memory

    def describe(self) -> Dict[str, Any]:
        """Get model description and conversation history."""
        conversation_usage = self.token_tracker.get_conversation_usage(self.conversation_id)
        total_usage = self.token_tracker.get_total_usage()
        
        return {
            "history": self.history_,
            "conversation_tokens": conversation_usage,
            "total_tokens_across_all_conversations": total_usage,
        }

    #@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def assistant_chat(self, message: str) -> str:
        """Send a message and get response using LangGraph."""
        if not self.app:
            raise RuntimeError(
                "Conversation not initialized. Call initialize_conversation first."
            )

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(content=message)]
        response = self.app.invoke({"messages": input_messages}, config)

        ai_message = response["messages"][-1]
        input_tokens = output_tokens = 0
        
        if hasattr(ai_message, "usage_metadata"):
            usage = ai_message.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            # Update both instance and global tracking
            self.input_tokens_ += input_tokens
            self.output_tokens_ += output_tokens
            self.token_tracker.add_tokens(
                self.conversation_id,
                input_tokens,
                output_tokens
            )

        self.history_.append(
            {
                "input": message,
                "output": ai_message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

        return ai_message.content

    async def astream(self, message: str):
        """Stream responses using LangGraph."""
        if not self.app:
            raise RuntimeError(
                "Conversation not initialized. Call initialize_conversation first."
            )

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(content=message)]

        async for chunk, metadata in self.app.stream(
            {"messages": input_messages}, config, stream_mode="messages"
        ):
            if isinstance(chunk, AIMessage):
                yield chunk.content


class AssistantChatOpenAI(ChatOpenAI, BaseAssistantChat):
    """OpenAI chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_name, "proxy": self.openai_proxy}


class AssistantChatBedrock(ChatBedrock, BaseAssistantChat):
    """Bedrock chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_id}


class ChatLLMFactory:
    """Factory class for creating chat models with LangGraph support."""
    
    @staticmethod
    def get_total_token_usage() -> Dict[str, int]:
        """Get total token usage across all conversations without creating a new LLM instance."""
        return GlobalTokenTracker().get_total_usage()

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
            logger.error(f"Error fetching OpenAI models: {e}")
            return []

    @staticmethod
    def get_bedrock_models() -> List[str]:
        try:
            bedrock = boto3.client("bedrock", region_name="us-west-2")
            response = bedrock.list_foundation_models()
            return [model["modelId"] for model in response["modelSummaries"]]
        except Exception as e:
            logger.error(f"Error fetching Bedrock models: {e}")
            return []

    @classmethod
    def get_chat_model(cls, config: DictConfig) -> BaseAssistantChat:
        """Get a configured chat model instance using LangGraph patterns."""
        provider = config.provider
        model = config.model

        valid_providers = ["openai", "bedrock"]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {provider}. Must be one of {valid_providers}"
            )

        valid_models = (
            cls.get_openai_models()
            if provider == "openai"
            else cls.get_bedrock_models()
        )
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model} for provider {provider}")

        if model not in WHITE_LIST_LLM:
            logger.warning(f"Model {model} is not on the white list: {WHITE_LIST_LLM}")

        if provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OpenAI API key not found in environment")

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
