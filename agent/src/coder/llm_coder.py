import logging
from typing import Dict, Optional
from omegaconf import DictConfig

from .utils import extract_script
from ..constants import VALID_CODING_LANGUAGES
from ..llm import ChatLLMFactory

logger = logging.getLogger(__name__)

class LLMCoder:
    """Class to handle code generation using LLM models."""
    
    def __init__(self, llm_config: DictConfig):
        """Initialize with LLM configuration.
        
        Args:
            llm_config: Configuration for the LLM model
        """
        self.llm_config = llm_config
        self.multi_turn = llm_config.multi_turn
        if self.multi_turn:
            self.llm = ChatLLMFactory.get_chat_model(llm_config)
        
    def __call__(
        self,
        prompt: str,
        language: str,
    ) -> Dict[str, str]:
        """Generate code using LLM based on prompt.
        
        Args:
            prompt: The coding prompt
            language: Target programming language
            
        Returns:
            Dictionary containing full response, language, extracted code
        """
        if not self.multi_turn:
            # create a new session if not multi_turn
            self.llm = ChatLLMFactory.get_chat_model(self.llm_config)

        if language not in VALID_CODING_LANGUAGES:
            raise ValueError(f"Language must be one of {VALID_CODING_LANGUAGES}")

        # Get response from LLM
        response = self.llm.assistant_chat(prompt)
        
        # Extract code from response
        code_script = extract_script(response, language.lower())
        
        return {
            "response": response,
            "language": language.lower(),
            "code_script": code_script,
        }
