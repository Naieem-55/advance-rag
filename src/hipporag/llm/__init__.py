import os
from copy import deepcopy

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .base import BaseLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    """Lazy import to avoid loading unnecessary dependencies on Windows."""

    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        from .bedrock_llm import BedrockLLM
        return BedrockLLM(config)

    if config.llm_name.startswith('Transformers/'):
        from .transformers_llm import TransformersLLM
        return TransformersLLM(config)

    if config.llm_name.startswith('gemini/'):
        from .gemini_llm import GeminiLLM
        return GeminiLLM.from_experiment_config(config)

    from .openai_gpt import CacheOpenAI
    return CacheOpenAI.from_experiment_config(config)


def _get_llm_for_task(config: BaseConfig, llm_name: str, llm_base_url: str):
    """Create an LLM instance for a specific task with custom model and base URL."""

    # Create a modified config for this specific LLM
    task_config = deepcopy(config)
    task_config.llm_name = llm_name
    task_config.llm_base_url = llm_base_url

    if llm_base_url is not None and 'localhost' in llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if llm_name.startswith('bedrock'):
        from .bedrock_llm import BedrockLLM
        return BedrockLLM(task_config)

    if llm_name.startswith('Transformers/'):
        from .transformers_llm import TransformersLLM
        return TransformersLLM(task_config)

    if llm_name.startswith('gemini/'):
        from .gemini_llm import GeminiLLM
        return GeminiLLM.from_experiment_config(task_config)

    from .openai_gpt import CacheOpenAI
    return CacheOpenAI.from_experiment_config(task_config)
    