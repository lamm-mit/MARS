"""LLM client interface and implementations."""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def strip_after_message_marker(text: str) -> str:
    """Strip text after message marker for gpt-oss compatibility."""
    marker = "final<|message|>"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    return text.strip()


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        pass


class GPTOSSClient(LLMClient):
    """GPT-OSS LLM client for local server using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 40000,
        timeout: int = 1200,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        self.client = OpenAI(api_key="NULL", base_url=base_url, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        try:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            result = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )

            text = result.choices[0].message.content
            return strip_after_message_marker(text)

        except Exception as e:
            error_msg = str(e)
            if "context_length" in error_msg.lower() or "maximum context length" in error_msg.lower():
                raise RuntimeError(
                    f"Context length exceeded: {error_msg}. "
                    f"Reduce input size or adjust max_input_tokens in config."
                ) from e
            logger.error(f"GPT-OSS API call failed: {e}", exc_info=True)
            raise RuntimeError(f"GPT-OSS API call failed: {e}") from e
