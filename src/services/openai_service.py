"""
OpenAI API service with circuit breaker and retry logic
Handles all interactions with OpenAI API including chat completions and image analysis
"""

import os
import base64
from typing import Dict, List, Optional, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import OpenAI, OpenAIError
from src.utils.logging_utils import get_logger
from src.utils.constants import (
    MAX_RETRY_ATTEMPTS,
    RETRY_WAIT_MIN_SECONDS,
    RETRY_WAIT_MAX_SECONDS,
    RETRY_WAIT_EXPONENTIAL_MULTIPLIER,
    OPENAI_MODEL_DEFAULT,
    OPENAI_MODEL_VISION,
    OPENAI_TEMPERATURE_DEFAULT
)

logger = get_logger("openai_service")


class OpenAIService:
    """
    Service for OpenAI API interactions with retry logic and circuit breaker pattern
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI service

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.error("openai_api_key_missing")
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=self.api_key)
        logger.info("openai_service_initialized")

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=RETRY_WAIT_EXPONENTIAL_MULTIPLIER,
            min=RETRY_WAIT_MIN_SECONDS,
            max=RETRY_WAIT_MAX_SECONDS
        ),
        retry=retry_if_exception_type((OpenAIError, ConnectionError, TimeoutError)),
        reraise=True
    )
    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Create a chat completion with retry logic and circuit breaker

        Args:
            messages: List of message dicts with role and content
            model: OpenAI model to use (defaults to OPENAI_MODEL_DEFAULT)
            temperature: Sampling temperature
            tools: Function calling tools
            tool_choice: Tool choice strategy
            response_format: Response format (e.g., {"type": "json_object"})
            **kwargs: Additional arguments to pass to the API

        Returns:
            OpenAI chat completion response

        Raises:
            OpenAIError: If API call fails after retries
        """
        model = model or os.getenv("OPENAI_MODEL_DEFAULT", OPENAI_MODEL_DEFAULT)
        temperature = temperature if temperature is not None else OPENAI_TEMPERATURE_DEFAULT

        try:
            logger.info(
                "openai_chat_completion_request",
                model=model,
                message_count=len(messages),
                has_tools=bool(tools)
            )

            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = tool_choice

            if response_format:
                api_params["response_format"] = response_format

            response = self.client.chat.completions.create(**api_params)

            logger.info(
                "openai_chat_completion_success",
                model=model,
                finish_reason=response.choices[0].finish_reason if response.choices else None
            )

            return response

        except OpenAIError as e:
            logger.error(
                "openai_api_error",
                error=str(e),
                error_type=type(e).__name__,
                model=model
            )
            raise
        except Exception as e:
            logger.error(
                "openai_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                model=model
            )
            raise

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=RETRY_WAIT_EXPONENTIAL_MULTIPLIER,
            min=RETRY_WAIT_MIN_SECONDS,
            max=RETRY_WAIT_MAX_SECONDS
        ),
        retry=retry_if_exception_type((OpenAIError, ConnectionError, TimeoutError)),
        reraise=True
    )
    def analyze_image(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str = "Ανέλυσε την εικόνα του τατουάζ.",
        model: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Analyze an image using OpenAI vision model with retry logic

        Args:
            image_path: Path to the image file
            system_prompt: System prompt for the analysis
            user_prompt: User prompt
            model: Vision model to use (defaults to OPENAI_MODEL_VISION)
            temperature: Sampling temperature

        Returns:
            Analysis text from the model

        Raises:
            FileNotFoundError: If image file doesn't exist
            OpenAIError: If API call fails after retries
        """
        model = model or os.getenv("OPENAI_MODEL_VISION", OPENAI_MODEL_VISION)

        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            logger.info(
                "openai_image_analysis_request",
                model=model,
                image_path=image_path
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                            }
                        ]
                    }
                ],
                temperature=temperature
            )

            result = response.choices[0].message.content.strip()

            logger.info(
                "openai_image_analysis_success",
                model=model,
                result_length=len(result)
            )

            return result

        except FileNotFoundError as e:
            logger.error("image_file_not_found", image_path=image_path)
            raise
        except OpenAIError as e:
            logger.error(
                "openai_vision_api_error",
                error=str(e),
                error_type=type(e).__name__,
                model=model
            )
            raise
        except Exception as e:
            logger.error(
                "openai_vision_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                model=model
            )
            raise

    def get_text_response(self, response: Any) -> str:
        """
        Extract text content from OpenAI response safely

        Args:
            response: OpenAI chat completion response

        Returns:
            Text content or error message
        """
        try:
            if response and response.choices:
                content = response.choices[0].message.content
                return content.strip() if content else "⚠️ Empty response"
            return "⚠️ Invalid response format"
        except Exception as e:
            logger.error("failed_to_extract_response", error=str(e))
            return "⚠️ Error extracting response"
