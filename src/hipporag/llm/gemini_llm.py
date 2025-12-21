"""
Gemini LLM integration for HippoRAG using LiteLLM
"""

import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

from filelock import FileLock
import litellm
from tenacity import retry, stop_after_attempt, wait_fixed

from .base import BaseLLM, LLMConfig
from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                return message, metadata, True

        result = func(self, *args, **kwargs)
        message, metadata = result

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper


class GeminiLLM(BaseLLM):
    """Gemini LLM using LiteLLM for API calls."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "GeminiLLM":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir: str, global_config: BaseConfig,
                 cache_filename: str = None, **kwargs) -> None:
        super().__init__(global_config)

        self.cache_dir = cache_dir
        self.global_config = global_config
        self.llm_name = global_config.llm_name

        # Verify Gemini API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        self.max_retries = kwargs.get("max_retries", 3)

        logger.info(f"Initialized Gemini LLM: {self.llm_name}")

    def _init_llm_config(self) -> None:
        """Initialize LLM configuration."""
        config_dict = self.global_config.__dict__.copy()

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['generate_params'] = {
            "model": self.global_config.llm_name,
            "max_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
        }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[str, dict]:
        """
        Run inference using Gemini via LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Tuple of (response_message, metadata)
        """
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages

        logger.debug(f"Calling Gemini API with model: {params['model']}")

        try:
            response = litellm.completion(**params)

            # Debug: Log the full response structure
            logger.debug(f"Full Gemini response: {response}")
            logger.debug(f"Response message: {response.choices[0].message}")
            logger.debug(f"Response message dict: {response.choices[0].message.__dict__ if hasattr(response.choices[0].message, '__dict__') else 'no dict'}")

            # Handle different response structures from Gemini 3
            message = response.choices[0].message
            response_content = message.content

            # If content is None or not a string, try to extract from different fields
            if response_content is None:
                # Check provider_specific_fields for Gemini 3
                if hasattr(message, 'provider_specific_fields') and message.provider_specific_fields:
                    psf = message.provider_specific_fields
                    if isinstance(psf, dict):
                        # Try to find text content in provider specific fields
                        if 'parts' in psf:
                            parts = psf['parts']
                            text_parts = [p.get('text', '') for p in parts if isinstance(p, dict) and 'text' in p]
                            response_message = ' '.join(text_parts)
                        elif 'text' in psf:
                            response_message = psf['text']
                        else:
                            response_message = str(psf)
                    else:
                        response_message = str(psf)
                # Try model_extra
                elif hasattr(message, 'model_extra') and message.model_extra:
                    extra = message.model_extra
                    if isinstance(extra, dict) and 'parts' in extra:
                        parts = extra['parts']
                        text_parts = [p.get('text', '') for p in parts if isinstance(p, dict) and 'text' in p]
                        response_message = ' '.join(text_parts)
                    else:
                        response_message = str(extra)
                # Try reasoning_content
                elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                    response_message = str(message.reasoning_content)
                elif hasattr(message, 'tool_calls') and message.tool_calls:
                    response_message = str(message.tool_calls)
                else:
                    # Print all available attributes for debugging
                    logger.warning(f"Content is None. Message attributes: {dir(message)}")
                    logger.warning(f"Message model_dump: {message.model_dump() if hasattr(message, 'model_dump') else 'N/A'}")
                    response_message = "No response content available"
            elif isinstance(response_content, list):
                # If content is a list (e.g., multi-part response), join text parts
                text_parts = []
                for part in response_content:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                    elif isinstance(part, str):
                        text_parts.append(part)
                    else:
                        text_parts.append(str(part))
                response_message = ' '.join(text_parts)
            elif isinstance(response_content, str):
                response_message = response_content
            else:
                response_message = str(response_content)

            metadata = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason,
            }

            return response_message, metadata

        except Exception as e:
            logger.error(f"Gemini inference error: {e}")
            raise
