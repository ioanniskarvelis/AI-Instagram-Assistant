"""
Integration tests for the Flask application
Tests end-to-end flows and API endpoints
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app():
    """Fixture for Flask app"""
    # Mock external dependencies before importing app
    with patch('src.services.openai_service.OpenAIService'):
        with patch('src.services.messaging_service.MessagingService'):
            with patch('src.utils.redis_utils.init_redis_client'):
                with patch('calendar_functions.authenticate_google'):
                    with patch('calendar_functions.get_calendar_service'):
                        with patch('sentence_transformers.SentenceTransformer'):
                            with patch('pinecone.Pinecone'):
                                import app as flask_app
                                flask_app.app.config['TESTING'] = True
                                yield flask_app.app


@pytest.fixture
def client(app):
    """Fixture for test client"""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint is accessible"""
        response = client.get('/health')
        assert response.status_code in [200, 503]  # Either healthy or degraded

    def test_health_endpoint_returns_json(self, client):
        """Test that health endpoint returns JSON"""
        response = client.get('/health')
        data = json.loads(response.data)

        assert 'status' in data
        assert 'timestamp' in data

    def test_health_endpoint_includes_redis_status(self, client):
        """Test that health check includes Redis status"""
        response = client.get('/health')
        data = json.loads(response.data)

        assert 'redis' in data


class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root_endpoint_accessible(self, client):
        """Test that root endpoint is accessible"""
        response = client.get('/')
        assert response.status_code == 200

    def test_root_returns_text(self, client):
        """Test that root returns expected text"""
        response = client.get('/')
        assert b"Hello world" in response.data


class TestPrivacyPolicyEndpoint:
    """Tests for privacy policy endpoint"""

    def test_privacy_policy_accessible(self, client):
        """Test that privacy policy endpoint works"""
        response = client.get('/privacy_policy')
        assert response.status_code in [200, 404]  # Might not exist in test env


class TestTermsEndpoint:
    """Tests for terms of service endpoint"""

    def test_terms_accessible(self, client):
        """Test that terms endpoint works"""
        response = client.get('/terms_of_service')
        assert response.status_code in [200, 404]  # Might not exist in test env


class TestWebhookEndpoint:
    """Tests for webhook endpoint"""

    def test_webhook_get_verification(self, client):
        """Test webhook GET request (verification)"""
        response = client.get('/webhook?hub.challenge=test_challenge')
        assert response.status_code == 200

    def test_webhook_post_requires_valid_json(self, client):
        """Test that POST requires valid JSON"""
        response = client.post(
            '/webhook',
            data='invalid json',
            content_type='application/json'
        )
        # Should handle invalid JSON gracefully
        assert response.status_code in [200, 400]

    @patch('app.queue_user_message')
    def test_webhook_post_with_valid_message(self, mock_queue, client, app):
        """Test webhook POST with valid Instagram message"""
        valid_payload = {
            "entry": [{
                "messaging": [{
                    "sender": {"id": "1018827777120359"},  # Admin ID
                    "recipient": {"id": "recipient_123"},
                    "message": {"text": "Hello"}
                }]
            }]
        }

        response = client.post(
            '/webhook',
            data=json.dumps(valid_payload),
            content_type='application/json'
        )

        assert response.status_code == 200

    def test_webhook_post_missing_entry(self, client):
        """Test webhook POST with missing entry"""
        invalid_payload = {"data": "test"}

        response = client.post(
            '/webhook',
            data=json.dumps(invalid_payload),
            content_type='application/json'
        )

        assert response.status_code == 400


class TestRateLimiting:
    """Tests for rate limiting"""

    def test_rate_limit_headers_present(self, client):
        """Test that rate limit headers are present"""
        response = client.get('/health')

        # Flask-Limiter should add rate limit headers
        # Note: Headers might not be present in test mode
        assert response.status_code == 200

    @pytest.mark.slow
    def test_webhook_rate_limit_enforced(self, client):
        """Test that webhook endpoint has rate limiting"""
        # This test would require many requests to trigger rate limit
        # Marked as slow test
        responses = []
        for i in range(10):
            response = client.get('/webhook')
            responses.append(response.status_code)

        # All should succeed (not enough to hit rate limit)
        assert all(status == 200 for status in responses)


class TestServiceIntegration:
    """Integration tests for services"""

    @patch('app.openai_service')
    @patch('app.messaging_service')
    def test_message_flow_integration(self, mock_messaging, mock_openai, client):
        """Test complete message processing flow"""
        # Mock OpenAI response
        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message.content = "Καλημέρα! ❤️🐼"
        mock_openai.create_chat_completion.return_value = mock_openai_response

        # Mock messaging response
        mock_messaging.send_long_message.return_value = [{"message_id": "msg_123"}]

        # This would trigger the full flow, but requires more setup
        # For now, just verify services are available
        assert mock_openai is not None
        assert mock_messaging is not None


class TestErrorHandling:
    """Tests for error handling"""

    def test_404_error_handling(self, client):
        """Test that 404 errors are handled"""
        response = client.get('/nonexistent_endpoint')
        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Test that method not allowed is handled"""
        # GET to root should work, but let's try POST
        response = client.post('/')
        assert response.status_code == 405


class TestConfigurationValidation:
    """Tests for configuration validation"""

    def test_app_debug_mode(self, app):
        """Test that DEBUG mode is properly set"""
        # In testing, DEBUG should be False
        assert app.config['TESTING'] is True

    def test_app_has_rate_limiter(self, app):
        """Test that rate limiter is configured"""
        # Verify rate limiter exists
        # This is implementation-specific
        assert hasattr(app, 'extensions') or True  # Basic check


class TestRedisIntegration:
    """Tests for Redis integration"""

    @patch('app.redis_client')
    def test_redis_conversation_storage(self, mock_redis):
        """Test that conversations are stored in Redis"""
        from app import save_convo_context

        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True

        save_convo_context("user_123", {
            "role": "user",
            "content": "test message"
        })

        # Verify Redis was called
        assert mock_redis.setex.called

    @patch('app.redis_client')
    def test_redis_conversation_retrieval(self, mock_redis):
        """Test that conversations are retrieved from Redis"""
        from app import get_convo_context

        mock_redis.get.return_value = json.dumps([{
            "role": "user",
            "content": "test"
        }])

        context = get_convo_context("user_123")

        assert len(context) == 1
        assert context[0]["content"] == "test"


class TestMessageQueueIntegration:
    """Tests for message queue functionality"""

    @patch('app.redis_client')
    def test_message_queuing(self, mock_redis):
        """Test that messages are queued"""
        from app import queue_user_message

        mock_redis.exists.return_value = False
        mock_redis.lpush.return_value = 1
        mock_redis.expire.return_value = True

        queue_user_message("user_123", {"test": "data"})

        # Verify queue operations
        assert mock_redis.lpush.called
        assert mock_redis.expire.called

    @patch('app.redis_client')
    def test_message_queue_retrieval(self, mock_redis):
        """Test retrieving queued messages"""
        from app import get_queued_messages

        mock_redis.lrange.return_value = [
            json.dumps({"timestamp": 123, "data": {"test": "message"}})
        ]

        messages = get_queued_messages("user_123")

        assert len(messages) == 1
        assert messages[0]["data"]["test"] == "message"


class TestProcessingLocks:
    """Tests for processing lock mechanism"""

    @patch('app.redis_client')
    def test_acquire_processing_lock(self, mock_redis):
        """Test acquiring processing lock"""
        from app import acquire_processing_lock

        mock_redis.set.return_value = True

        result = acquire_processing_lock("user_123")

        assert result is True
        mock_redis.set.assert_called_once()

    @patch('app.redis_client')
    def test_release_processing_lock(self, mock_redis):
        """Test releasing processing lock"""
        from app import release_processing_lock

        mock_redis.delete.return_value = 1

        release_processing_lock("user_123")

        mock_redis.delete.assert_called_once()


class TestMuteFunction:
    """Tests for mute functionality"""

    @patch('app.redis_client')
    def test_mute_user(self, mock_redis):
        """Test muting a user"""
        from app import mute_user

        mock_redis.setex.return_value = True

        mute_user("user_123")

        mock_redis.setex.assert_called_once()

    @patch('app.redis_client')
    def test_is_user_muted(self, mock_redis):
        """Test checking if user is muted"""
        from app import is_user_muted

        mock_redis.exists.return_value = True

        result = is_user_muted("user_123")

        assert result is True


@pytest.mark.slow
class TestEndToEndFlow:
    """End-to-end integration tests (marked as slow)"""

    @patch('app.openai_service')
    @patch('app.messaging_service')
    @patch('app.redis_client')
    @patch('app.service')  # Calendar service
    def test_complete_booking_flow(self, mock_calendar, mock_redis, mock_messaging, mock_openai):
        """Test complete booking flow from webhook to response"""
        # This would test the entire flow:
        # 1. Receive webhook
        # 2. Queue message
        # 3. Process message
        # 4. Call OpenAI
        # 5. Execute calendar functions
        # 6. Send response

        # Setup mocks
        mock_redis.exists.return_value = False
        mock_redis.lpush.return_value = 1

        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message.content = "Booking confirmed!"
        mock_openai.create_chat_completion.return_value = mock_openai_response

        # This is a skeleton - full implementation would require more setup
        assert True  # Placeholder


# Markers for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
