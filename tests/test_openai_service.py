"""
Unit tests for OpenAIService
Tests circuit breaker, retry logic, and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAIError
from src.services.openai_service import OpenAIService


class TestOpenAIServiceInitialization:
    """Tests for OpenAIService initialization"""

    def test_init_with_api_key(self):
        """Test initialization with API key"""
        service = OpenAIService(api_key="test-key")
        assert service.api_key == "test-key"
        assert service.client is not None

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable"""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        service = OpenAIService()
        assert service.api_key == "env-test-key"

    def test_init_without_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises ValueError"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIService()


class TestChatCompletion:
    """Tests for chat completion with retry logic"""

    @pytest.fixture
    def service(self):
        """Fixture for OpenAIService"""
        return OpenAIService(api_key="test-key")

    @pytest.fixture
    def mock_response(self):
        """Fixture for mock OpenAI response"""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Test response"
        response.choices[0].finish_reason = "stop"
        return response

    def test_successful_chat_completion(self, service, mock_response):
        """Test successful chat completion"""
        with patch.object(service.client.chat.completions, 'create', return_value=mock_response):
            result = service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o"
            )

            assert result == mock_response
            assert result.choices[0].message.content == "Test response"

    def test_chat_completion_with_tools(self, service, mock_response):
        """Test chat completion with function calling tools"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function"
                }
            }
        ]

        with patch.object(service.client.chat.completions, 'create', return_value=mock_response):
            result = service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                tools=tools,
                tool_choice="auto"
            )

            assert result == mock_response

    def test_chat_completion_with_json_format(self, service, mock_response):
        """Test chat completion with JSON response format"""
        with patch.object(service.client.chat.completions, 'create', return_value=mock_response):
            result = service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                response_format={"type": "json_object"}
            )

            assert result == mock_response

    def test_retry_on_openai_error(self, service, mock_response):
        """Test that OpenAI errors trigger retry"""
        mock_create = Mock(side_effect=[
            OpenAIError("Rate limit"),
            OpenAIError("Temporary failure"),
            mock_response  # Success on third try
        ])

        with patch.object(service.client.chat.completions, 'create', mock_create):
            result = service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

            # Should have retried and eventually succeeded
            assert result == mock_response
            assert mock_create.call_count == 3

    def test_retry_exhausted_raises_error(self, service):
        """Test that exhausted retries raise the error"""
        mock_create = Mock(side_effect=OpenAIError("Persistent error"))

        with patch.object(service.client.chat.completions, 'create', mock_create):
            with pytest.raises(OpenAIError, match="Persistent error"):
                service.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}]
                )

            # Should have tried 3 times (max retries)
            assert mock_create.call_count == 3

    def test_connection_error_triggers_retry(self, service, mock_response):
        """Test that connection errors trigger retry"""
        mock_create = Mock(side_effect=[
            ConnectionError("Network issue"),
            mock_response  # Success on second try
        ])

        with patch.object(service.client.chat.completions, 'create', mock_create):
            result = service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert result == mock_response
            assert mock_create.call_count == 2

    def test_custom_temperature(self, service, mock_response):
        """Test custom temperature parameter"""
        with patch.object(service.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.5
            )

            # Verify temperature was passed
            call_args = mock_create.call_args[1]
            assert call_args['temperature'] == 0.5

    def test_custom_model(self, service, mock_response):
        """Test custom model parameter"""
        with patch.object(service.client.chat.completions, 'create', return_value=mock_response) as mock_create:
            service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o-mini"
            )

            call_args = mock_create.call_args[1]
            assert call_args['model'] == "gpt-4o-mini"


class TestImageAnalysis:
    """Tests for image analysis with vision model"""

    @pytest.fixture
    def service(self):
        """Fixture for OpenAIService"""
        return OpenAIService(api_key="test-key")

    @pytest.fixture
    def mock_vision_response(self):
        """Fixture for mock vision API response"""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Fine line tattoo | h=5 | w=5 | ink=0.10 | D=1.14"
        return response

    @pytest.fixture
    def temp_image(self, tmp_path):
        """Fixture for temporary test image"""
        image_path = tmp_path / "test_tattoo.jpg"
        image_path.write_bytes(b"fake image data")
        return str(image_path)

    def test_successful_image_analysis(self, service, temp_image, mock_vision_response):
        """Test successful image analysis"""
        with patch.object(service.client.chat.completions, 'create', return_value=mock_vision_response):
            result = service.analyze_image(
                image_path=temp_image,
                system_prompt="Analyze this tattoo",
                user_prompt="What do you see?"
            )

            assert "Fine line tattoo" in result
            assert "h=5" in result

    def test_image_not_found_raises_error(self, service):
        """Test that missing image file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            service.analyze_image(
                image_path="/nonexistent/image.jpg",
                system_prompt="Analyze this",
                user_prompt="What do you see?"
            )

    def test_image_analysis_retry_on_error(self, service, temp_image, mock_vision_response):
        """Test retry logic for image analysis"""
        mock_create = Mock(side_effect=[
            OpenAIError("API error"),
            mock_vision_response  # Success on second try
        ])

        with patch.object(service.client.chat.completions, 'create', mock_create):
            result = service.analyze_image(
                image_path=temp_image,
                system_prompt="Analyze this",
                user_prompt="What do you see?"
            )

            assert result == mock_vision_response.choices[0].message.content.strip()
            assert mock_create.call_count == 2

    def test_custom_vision_model(self, service, temp_image, mock_vision_response):
        """Test custom vision model parameter"""
        with patch.object(service.client.chat.completions, 'create', return_value=mock_vision_response) as mock_create:
            service.analyze_image(
                image_path=temp_image,
                system_prompt="Analyze this",
                user_prompt="What do you see?",
                model="gpt-4o"
            )

            call_args = mock_create.call_args[1]
            assert call_args['model'] == "gpt-4o"

    def test_custom_temperature_for_vision(self, service, temp_image, mock_vision_response):
        """Test custom temperature for vision analysis"""
        with patch.object(service.client.chat.completions, 'create', return_value=mock_vision_response) as mock_create:
            service.analyze_image(
                image_path=temp_image,
                system_prompt="Analyze this",
                user_prompt="What do you see?",
                temperature=0.5
            )

            call_args = mock_create.call_args[1]
            assert call_args['temperature'] == 0.5


class TestGetTextResponse:
    """Tests for safe response extraction"""

    @pytest.fixture
    def service(self):
        """Fixture for OpenAIService"""
        return OpenAIService(api_key="test-key")

    def test_extract_valid_response(self, service):
        """Test extracting text from valid response"""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Test content"

        result = service.get_text_response(response)
        assert result == "Test content"

    def test_extract_response_with_whitespace(self, service):
        """Test that whitespace is stripped"""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "  Test content  \n"

        result = service.get_text_response(response)
        assert result == "Test content"

    def test_extract_empty_content(self, service):
        """Test handling of empty content"""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = None

        result = service.get_text_response(response)
        assert "Empty response" in result

    def test_extract_no_choices(self, service):
        """Test handling of response with no choices"""
        response = Mock()
        response.choices = []

        result = service.get_text_response(response)
        assert "Invalid response" in result

    def test_extract_none_response(self, service):
        """Test handling of None response"""
        result = service.get_text_response(None)
        assert "Invalid response" in result


class TestErrorHandling:
    """Tests for error handling and logging"""

    @pytest.fixture
    def service(self):
        """Fixture for OpenAIService"""
        return OpenAIService(api_key="test-key")

    def test_unexpected_error_handling(self, service):
        """Test handling of unexpected errors"""
        mock_create = Mock(side_effect=ValueError("Unexpected error"))

        with patch.object(service.client.chat.completions, 'create', mock_create):
            with pytest.raises(ValueError, match="Unexpected error"):
                service.create_chat_completion(
                    messages=[{"role": "user", "content": "Hello"}]
                )

    def test_timeout_error_triggers_retry(self, service, mocker):
        """Test that timeout errors trigger retry"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success"

        mock_create = Mock(side_effect=[
            TimeoutError("Request timeout"),
            mock_response
        ])

        with patch.object(service.client.chat.completions, 'create', mock_create):
            result = service.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert result == mock_response
            assert mock_create.call_count == 2
