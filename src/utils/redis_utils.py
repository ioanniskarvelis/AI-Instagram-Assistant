"""
Redis utilities and connection management
Provides robust Redis client initialization with error handling
"""

import os
import redis
from typing import Optional
from src.utils.logging_utils import get_logger

logger = get_logger("redis")


def init_redis_client() -> redis.Redis:
    """
    Initialize the Redis client using environment variables or Redis URL

    Returns:
        Configured Redis client

    Raises:
        ValueError: If neither REDIS_URL nor REDIS_HOST is configured
        Exception: If Redis connection fails
    """
    # Load from environment
    redis_url = os.getenv('REDIS_URL', '')
    redis_host = os.getenv('REDIS_HOST', '')
    redis_port = int(os.getenv('REDIS_PORT', '6379')) if os.getenv('REDIS_PORT') else None
    redis_username = os.getenv('REDIS_USERNAME', '')
    redis_password = os.getenv('REDIS_PASSWORD', '')

    try:
        # Option 1: Use Redis URL (preferred)
        if redis_url:
            logger.info("connecting_to_redis", method="url")
            return redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=30,
                socket_connect_timeout=30,
                retry_on_timeout=True,
                retry_on_error=[redis.exceptions.ConnectionError, redis.exceptions.TimeoutError],
                health_check_interval=30
            )

        # Option 2: Use host/port configuration
        elif redis_host:
            logger.info("connecting_to_redis", method="host_port", host=redis_host, port=redis_port)
            redis_config = {
                'host': redis_host,
                'port': redis_port or 6379,
                'username': redis_username or None,
                'password': redis_password or None,
                'db': 0,
                'decode_responses': True,
                'socket_timeout': 30,
                'socket_connect_timeout': 30,
                'retry_on_timeout': True,
                'retry_on_error': [redis.exceptions.ConnectionError, redis.exceptions.TimeoutError],
                'health_check_interval': 30
            }

            # Enable SSL if REDIS_SSL is true
            if os.getenv('REDIS_SSL', 'false').lower() == 'true':
                redis_config['ssl'] = True
                redis_config['ssl_cert_reqs'] = None

            return redis.Redis(**redis_config)

        else:
            raise ValueError("Neither REDIS_URL nor REDIS_HOST is configured")

    except Exception as e:
        logger.error("redis_init_failed", error=str(e))
        raise


def test_redis_connection(client: redis.Redis) -> bool:
    """
    Test Redis connection and log status

    Args:
        client: Redis client to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Test basic connectivity
        client.ping()
        logger.info("redis_connection_successful")

        # Test basic operations
        test_key = "health_check"
        client.set(test_key, "ok", ex=10)
        value = client.get(test_key)
        client.delete(test_key)

        if value == "ok":
            logger.info("redis_operations_working")
            return True
        else:
            logger.error("redis_readwrite_test_failed")
            return False

    except redis.exceptions.ConnectionError as e:
        logger.error("redis_connection_failed", error=str(e))
        return False
    except Exception as e:
        logger.error("redis_test_failed", error=str(e))
        return False
