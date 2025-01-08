import logging
import os
from typing import Any, Dict, List, Sequence, Optional
import uuid

import boto3
from autogluon.assistant.constants import WHITE_LIST_LLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class BaseAssistantChat(BaseModel):
    """Base class for assistant chat models with LangGraph support."""
    
    # Configure Pydantic to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_tokens_: int = Field(default=0)
    output_tokens_: int = Field(default=0)
    graph: Optional[Any] = Field(default=None, exclude=True)
    app: Optional[Any] = Field(default=None, exclude=True)
    memory: Optional[Any] = Field(default=None, exclude=True)

    def initialize_conversation(
        self,
        llm: Any,
        system_prompt: str = "You are a technical assistant that excels at working on data science tasks.",
    ) -> None:
        """Initialize conversation using LangGraph."""
        # Create prompt template with message history
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])

        # Define the graph
        graph = StateGraph(state_schema=MessagesState)

        # Define the function that calls the model
        def call_model(state: MessagesState):
            # Create the full prompt with system message and history
            prompt_messages = prompt_template.invoke(state)
            # Get response from the model
            response = llm.invoke(prompt_messages)
            return {"messages": [response]}

        # Add the single node to the graph
        graph.add_edge(START, "model")
        graph.add_node("model", call_model)

        # Initialize memory
        memory = MemorySaver()
        
        # Compile the graph into a runnable application
        app = graph.compile(checkpointer=memory)

        # Store references using protected attributes
        self.graph = graph
        self.app = app
        self.memory = memory

    def describe(self) -> Dict[str, Any]:
        """Get model description and conversation history."""
        return {
            "history": self.history_,
            "prompt_tokens": self.input_tokens_,
            "completion_tokens": self.output_tokens_,
        }

    def assistant_chat(self, message: str) -> str:
        """Send a message and get response using LangGraph."""
        if not self.app:
            raise RuntimeError("Conversation not initialized. Call initialize_conversation first.")

        # Generate a thread ID for this conversation
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Prepare the input message
        input_messages = [HumanMessage(content=message)]

        # Invoke the graph
        response = self.app.invoke(
            {"messages": input_messages},
            config
        )

        # Update token counts if available
        ai_message = response["messages"][-1]
        if hasattr(ai_message, "usage_metadata"):
            usage = ai_message.usage_metadata
            self.input_tokens_ += usage.get("input_tokens", 0)
            self.output_tokens_ += usage.get("output_tokens", 0)

        # Record in history
        self.history_.append({
            "input": message,
            "output": ai_message.content,
            "prompt_tokens": self.input_tokens_,
            "completion_tokens": self.output_tokens_
        })

        return ai_message.content

    async def astream(self, message: str):
        """Stream responses using LangGraph."""
        if not self.app:
            raise RuntimeError("Conversation not initialized. Call initialize_conversation first.")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(content=message)]

        async for chunk, metadata in self.app.stream(
            {"messages": input_messages},
            config,
            stream_mode="messages"
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
        return {
            **base_desc,
            "model": self.model_name,
            "proxy": self.openai_proxy,
        }

class AssistantChatBedrock(ChatBedrock, BaseAssistantChat):
    """Bedrock chat model with LangGraph support."""

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
    """Factory class for creating chat models with LangGraph support."""

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

        # Validate provider
        valid_providers = ["openai", "bedrock"]
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

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