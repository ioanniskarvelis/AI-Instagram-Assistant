"""
Instagram messaging service
Handles all interactions with Instagram Graph API for sending messages
"""

import os
import requests
from typing import Dict, Any, List
from src.utils.logging_utils import get_logger
from src.utils.constants import MESSAGE_MAX_LENGTH

logger = get_logger("messaging_service")


class MessagingService:
    """
    Service for Instagram Graph API messaging operations
    """

    def __init__(self, access_token: str = None):
        """
        Initialize messaging service

        Args:
            access_token: Instagram User Access Token
        """
        self.access_token = access_token or os.getenv("IG_USER_ACCESS_TOKEN", "")
        if not self.access_token:
            logger.error("instagram_access_token_missing")
            raise ValueError("IG_USER_ACCESS_TOKEN environment variable is required")

        self.api_url = "https://graph.instagram.com/v22.0/me/messages"
        logger.info("messaging_service_initialized")

    def send_message(self, recipient_id: str, message_text: str) -> Dict[str, Any]:
        """
        Send a message to an Instagram user

        Args:
            recipient_id: Instagram user ID (PSID)
            message_text: Message text to send

        Returns:
            API response dict

        Raises:
            Exception: If API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "recipient": {
                "id": recipient_id
            },
            "message": {
                "text": message_text
            }
        }

        try:
            logger.info(
                "sending_instagram_message",
                recipient_id=recipient_id,
                message_length=len(message_text)
            )

            response = requests.post(self.api_url, headers=headers, json=payload)
            result = response.json()

            if response.status_code == 200:
                logger.info(
                    "instagram_message_sent",
                    recipient_id=recipient_id,
                    message_id=result.get("message_id")
                )
            else:
                logger.error(
                    "instagram_message_failed",
                    recipient_id=recipient_id,
                    status_code=response.status_code,
                    error=result.get("error")
                )

            return result

        except Exception as e:
            logger.error(
                "instagram_send_error",
                recipient_id=recipient_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return {"error": str(e)}

    def send_long_message(self, recipient_id: str, message_text: str) -> List[Dict[str, Any]]:
        """
        Send a long message by splitting it into chunks if needed

        Args:
            recipient_id: Instagram user ID
            message_text: Message text (may be longer than max length)

        Returns:
            List of API response dicts (one per chunk sent)
        """
        if len(message_text) <= MESSAGE_MAX_LENGTH:
            return [self.send_message(recipient_id, message_text)]

        logger.info(
            "splitting_long_message",
            recipient_id=recipient_id,
            total_length=len(message_text),
            max_length=MESSAGE_MAX_LENGTH
        )

        # Split into chunks without breaking words
        chunks = self._split_message(message_text, MESSAGE_MAX_LENGTH)
        responses = []

        for i, chunk in enumerate(chunks, 1):
            logger.info(
                "sending_message_chunk",
                recipient_id=recipient_id,
                chunk_number=i,
                total_chunks=len(chunks)
            )
            response = self.send_message(recipient_id, chunk)
            responses.append(response)

        return responses

    def _split_message(self, text: str, max_length: int) -> List[str]:
        """
        Split message into chunks at natural break points

        Args:
            text: Text to split
            max_length: Maximum chunk length

        Returns:
            List of text chunks
        """
        chunks = []

        while len(text) > max_length:
            # Try to split at newline
            split_at = text.rfind('\n', 0, max_length)
            if split_at == -1:
                # Try to split at space
                split_at = text.rfind(' ', 0, max_length)
            if split_at == -1:
                # No good split point, just split at max_length
                split_at = max_length

            chunks.append(text[:split_at].strip())
            text = text[split_at:].strip()

        if text:
            chunks.append(text)

        return chunks

    def download_image(self, image_url: str, user_id: str) -> str:
        """
        Download an image from Instagram

        Args:
            image_url: URL of the image
            user_id: User ID (for unique filename)

        Returns:
            Path to downloaded image file

        Raises:
            Exception: If download fails
        """
        import uuid

        # Define temp directory
        temp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(temp_dir, exist_ok=True)

        # Generate unique filename
        unique_id = uuid.uuid1().hex
        image_path = os.path.join(temp_dir, f"{user_id}_tattoo_{unique_id}.jpg")

        try:
            logger.info(
                "downloading_image",
                user_id=user_id,
                url=image_url[:50] + "..."
            )

            response = requests.get(image_url, timeout=30)

            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)

                logger.info(
                    "image_downloaded",
                    user_id=user_id,
                    path=image_path,
                    size_bytes=len(response.content)
                )

                return image_path
            else:
                raise Exception(f"Failed to download image. Status code: {response.status_code}")

        except Exception as e:
            logger.error(
                "image_download_failed",
                user_id=user_id,
                error=str(e)
            )
            raise
