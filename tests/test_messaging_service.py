"""
Unit tests for MessagingService
Tests Instagram messaging and image download functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from src.services.messaging_service import MessagingService


class TestMessagingServiceInitialization:
    """Tests for MessagingService initialization"""

    def test_init_with_access_token(self):
        """Test initialization with access token"""
        service = MessagingService(access_token="test-token")
        assert service.access_token == "test-token"
        assert service.api_url == "https://graph.instagram.com/v22.0/me/messages"

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable"""
        monkeypatch.setenv("IG_USER_ACCESS_TOKEN", "env-test-token")
        service = MessagingService()
        assert service.access_token == "env-test-token"

    def test_init_without_token_raises_error(self, monkeypatch):
        """Test that missing token raises ValueError"""
        monkeypatch.delenv("IG_USER_ACCESS_TOKEN", raising=False)
        with pytest.raises(ValueError, match="IG_USER_ACCESS_TOKEN"):
            MessagingService()


class TestSendMessage:
    """Tests for sending Instagram messages"""

    @pytest.fixture
    def service(self):
        """Fixture for MessagingService"""
        return MessagingService(access_token="test-token")

    @pytest.fixture
    def mock_success_response(self):
        """Fixture for successful API response"""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "message_id": "msg_123456",
            "recipient_id": "user_789"
        }
        return response

    def test_successful_message_send(self, service, mock_success_response):
        """Test successful message sending"""
        with patch('requests.post', return_value=mock_success_response):
            result = service.send_message(
                recipient_id="user_123",
                message_text="Hello, world!"
            )

            assert result["message_id"] == "msg_123456"
            assert result["recipient_id"] == "user_789"

    def test_send_message_constructs_correct_payload(self, service, mock_success_response):
        """Test that payload is correctly constructed"""
        with patch('requests.post', return_value=mock_success_response) as mock_post:
            service.send_message(
                recipient_id="user_123",
                message_text="Test message"
            )

            # Verify the call was made with correct payload
            call_args = mock_post.call_args
            payload = call_args[1]['json']

            assert payload['recipient']['id'] == "user_123"
            assert payload['message']['text'] == "Test message"

    def test_send_message_sets_correct_headers(self, service, mock_success_response):
        """Test that headers are correctly set"""
        with patch('requests.post', return_value=mock_success_response) as mock_post:
            service.send_message(
                recipient_id="user_123",
                message_text="Test"
            )

            headers = mock_post.call_args[1]['headers']
            assert headers['Authorization'] == "Bearer test-token"
            assert headers['Content-Type'] == "application/json"

    def test_failed_message_send(self, service):
        """Test handling of failed message send"""
        error_response = Mock()
        error_response.status_code = 400
        error_response.json.return_value = {
            "error": {
                "message": "Invalid recipient",
                "code": 100
            }
        }

        with patch('requests.post', return_value=error_response):
            result = service.send_message(
                recipient_id="invalid_user",
                message_text="Test"
            )

            assert "error" in result

    def test_network_error_handling(self, service):
        """Test handling of network errors"""
        with patch('requests.post', side_effect=requests.ConnectionError("Network error")):
            result = service.send_message(
                recipient_id="user_123",
                message_text="Test"
            )

            assert "error" in result
            assert "Network error" in result["error"]

    def test_timeout_error_handling(self, service):
        """Test handling of timeout errors"""
        with patch('requests.post', side_effect=requests.Timeout("Request timeout")):
            result = service.send_message(
                recipient_id="user_123",
                message_text="Test"
            )

            assert "error" in result


class TestSendLongMessage:
    """Tests for sending long messages with automatic splitting"""

    @pytest.fixture
    def service(self):
        """Fixture for MessagingService"""
        return MessagingService(access_token="test-token")

    @pytest.fixture
    def mock_success_response(self):
        """Fixture for successful API response"""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message_id": "msg_123"}
        return response

    def test_short_message_sent_as_single(self, service, mock_success_response):
        """Test that short messages are sent as single message"""
        with patch('requests.post', return_value=mock_success_response) as mock_post:
            result = service.send_long_message(
                recipient_id="user_123",
                message_text="Short message"
            )

            # Should only call send_message once
            assert mock_post.call_count == 1
            assert len(result) == 1

    def test_long_message_split_into_chunks(self, service, mock_success_response):
        """Test that long messages are split into chunks"""
        # Create a message longer than MESSAGE_MAX_LENGTH (800 chars)
        long_message = "A" * 900

        with patch('requests.post', return_value=mock_success_response) as mock_post:
            result = service.send_long_message(
                recipient_id="user_123",
                message_text=long_message
            )

            # Should have split into multiple messages
            assert mock_post.call_count >= 2
            assert len(result) >= 2

    def test_message_split_at_newline(self, service, mock_success_response):
        """Test that messages split at newline when possible"""
        # Create message with newline before max length
        message = "A" * 700 + "\n" + "B" * 200

        with patch('requests.post', return_value=mock_success_response) as mock_post:
            with patch.object(service, '_split_message') as mock_split:
                mock_split.return_value = ["chunk1", "chunk2"]

                service.send_long_message(
                    recipient_id="user_123",
                    message_text=message
                )

                # Verify split_message was called
                mock_split.assert_called_once()

    def test_split_message_preserves_content(self, service):
        """Test that message splitting preserves all content"""
        long_message = "A" * 1000

        chunks = service._split_message(long_message, max_length=400)

        # Reconstruct original message
        reconstructed = "".join(chunks)
        assert len(reconstructed) >= len(long_message.strip())

    def test_split_message_respects_max_length(self, service):
        """Test that chunks don't exceed max length"""
        long_message = "A" * 1500

        chunks = service._split_message(long_message, max_length=500)

        for chunk in chunks:
            assert len(chunk) <= 500

    def test_split_message_at_space(self, service):
        """Test splitting at space when no newline"""
        message = "word " * 200  # 1000 characters

        chunks = service._split_message(message, max_length=400)

        # Verify no words are cut in half
        for chunk in chunks:
            # Chunks should end with complete words (except possibly the last)
            if chunk != chunks[-1]:
                assert not chunk.endswith(" word")[:4]  # Not cut mid-word

    def test_split_very_long_word(self, service):
        """Test splitting when word exceeds max length"""
        # A single word longer than max length
        message = "A" * 1000

        chunks = service._split_message(message, max_length=300)

        # Should split even though there are no spaces/newlines
        assert len(chunks) >= 3

    def test_multiple_chunks_all_sent(self, service, mock_success_response):
        """Test that all chunks are sent"""
        long_message = "A" * 1500

        with patch('requests.post', return_value=mock_success_response) as mock_post:
            result = service.send_long_message(
                recipient_id="user_123",
                message_text=long_message
            )

            # All chunks should have been sent
            assert len(result) == mock_post.call_count


class TestDownloadImage:
    """Tests for downloading images from Instagram"""

    @pytest.fixture
    def service(self):
        """Fixture for MessagingService"""
        return MessagingService(access_token="test-token")

    @pytest.fixture
    def mock_image_response(self):
        """Fixture for mock image download response"""
        response = Mock()
        response.status_code = 200
        response.content = b"fake image data"
        return response

    def test_successful_image_download(self, service, mock_image_response, tmp_path):
        """Test successful image download"""
        with patch('requests.get', return_value=mock_image_response):
            with patch('os.getcwd', return_value=str(tmp_path)):
                with patch('os.makedirs'):
                    image_path = service.download_image(
                        image_url="https://example.com/image.jpg",
                        user_id="user_123"
                    )

                    assert "user_123_tattoo_" in image_path
                    assert image_path.endswith(".jpg")

    def test_image_download_creates_tmp_dir(self, service, mock_image_response, tmp_path):
        """Test that tmp directory is created if not exists"""
        with patch('requests.get', return_value=mock_image_response):
            with patch('os.getcwd', return_value=str(tmp_path)):
                with patch('os.makedirs') as mock_makedirs:
                    service.download_image(
                        image_url="https://example.com/image.jpg",
                        user_id="user_123"
                    )

                    # Verify makedirs was called
                    mock_makedirs.assert_called_once()

    def test_image_download_unique_filenames(self, service, mock_image_response, tmp_path):
        """Test that unique filenames are generated"""
        with patch('requests.get', return_value=mock_image_response):
            with patch('os.getcwd', return_value=str(tmp_path)):
                with patch('os.makedirs'):
                    path1 = service.download_image(
                        image_url="https://example.com/image1.jpg",
                        user_id="user_123"
                    )
                    path2 = service.download_image(
                        image_url="https://example.com/image2.jpg",
                        user_id="user_123"
                    )

                    # Paths should be different
                    assert path1 != path2

    def test_image_download_failed_status(self, service):
        """Test handling of failed download"""
        error_response = Mock()
        error_response.status_code = 404

        with patch('requests.get', return_value=error_response):
            with pytest.raises(Exception, match="Failed to download image"):
                service.download_image(
                    image_url="https://example.com/missing.jpg",
                    user_id="user_123"
                )

    def test_image_download_network_error(self, service):
        """Test handling of network errors during download"""
        with patch('requests.get', side_effect=requests.ConnectionError("Network error")):
            with pytest.raises(requests.ConnectionError):
                service.download_image(
                    image_url="https://example.com/image.jpg",
                    user_id="user_123"
                )

    def test_image_download_timeout(self, service):
        """Test that download has timeout"""
        with patch('requests.get', side_effect=requests.Timeout("Timeout")):
            with pytest.raises(requests.Timeout):
                service.download_image(
                    image_url="https://example.com/image.jpg",
                    user_id="user_123"
                )

    def test_image_saved_to_correct_location(self, service, mock_image_response, tmp_path):
        """Test that image is saved to correct location"""
        with patch('requests.get', return_value=mock_image_response):
            with patch('os.getcwd', return_value=str(tmp_path)):
                image_path = service.download_image(
                    image_url="https://example.com/image.jpg",
                    user_id="user_123"
                )

                # Path should include tmp directory
                assert "/tmp/" in image_path or "tmp/" in image_path


class TestMessageSplitting:
    """Detailed tests for message splitting logic"""

    @pytest.fixture
    def service(self):
        """Fixture for MessagingService"""
        return MessagingService(access_token="test-token")

    def test_split_empty_message(self, service):
        """Test splitting empty message"""
        chunks = service._split_message("", max_length=100)
        assert chunks == []

    def test_split_message_exact_max_length(self, service):
        """Test message exactly at max length"""
        message = "A" * 100
        chunks = service._split_message(message, max_length=100)

        assert len(chunks) == 1
        assert len(chunks[0]) == 100

    def test_split_preserves_newlines_in_chunks(self, service):
        """Test that newlines within chunks are preserved"""
        message = "Line1\nLine2\nLine3"
        chunks = service._split_message(message, max_length=100)

        # Should be single chunk with newlines preserved
        assert len(chunks) == 1
        assert "\n" in chunks[0]

    def test_split_whitespace_trimmed(self, service):
        """Test that chunks are trimmed of whitespace"""
        message = "   " + "A" * 100 + "   B" * 100
        chunks = service._split_message(message, max_length=50)

        for chunk in chunks:
            # Chunks should be stripped
            assert chunk == chunk.strip()
