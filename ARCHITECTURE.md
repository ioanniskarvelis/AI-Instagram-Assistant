# Architecture Documentation

## Overview

This document describes the modular architecture of the AI Instagram Assistant application. The codebase has been refactored from a monolithic structure into a well-organized modular design following best practices.

---

## Directory Structure

```
AI-Instagram-Assistant/
├── src/
│   ├── services/          # Business logic layer
│   │   ├── openai_service.py       # OpenAI API interactions with retry logic
│   │   └── messaging_service.py    # Instagram messaging operations
│   ├── utils/             # Utility modules
│   │   ├── logging_utils.py        # Structured logging with structlog
│   │   ├── redis_utils.py          # Redis connection management
│   │   └── constants.py            # Application constants
│   └── __init__.py
├── tests/                 # Test suite
│   ├── test_validation.py
│   ├── test_calendar_functions.py
│   └── conftest.py
├── prompts/               # AI prompts for different intents
├── app.py                 # Main Flask application
├── calendar_functions.py  # Google Calendar operations
├── validation.py          # Input validation utilities
└── requirements.txt       # Python dependencies

```

---

## Module Descriptions

### Services Layer (`src/services/`)

Business logic modules that handle external API interactions and core functionality.

#### `openai_service.py`
- **Purpose**: Manages all OpenAI API interactions
- **Features**:
  - Circuit breaker pattern with `tenacity` for automatic retries
  - Exponential backoff for failed requests
  - Structured logging of all API calls
  - Chat completions with function calling support
  - Image analysis using vision models
- **Key Methods**:
  - `create_chat_completion()` - Chat API with retry logic
  - `analyze_image()` - Vision API for tattoo image analysis
  - `get_text_response()` - Safe response extraction

**Example Usage**:
```python
from src.services.openai_service import OpenAIService

service = OpenAIService()
response = service.create_chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o"
)
```

#### `messaging_service.py`
- **Purpose**: Handles Instagram Graph API messaging
- **Features**:
  - Send single messages
  - Automatic message splitting for long texts
  - Image downloads from Instagram
  - Error handling and logging
- **Key Methods**:
  - `send_message()` - Send single message
  - `send_long_message()` - Handles long messages with smart splitting
  - `download_image()` - Download image attachments

**Example Usage**:
```python
from src.services.messaging_service import MessagingService

service = MessagingService()
service.send_long_message(
    recipient_id="123456",
    message_text="Long message..."
)
```

---

### Utils Layer (`src/utils/`)

Utility modules providing shared functionality across the application.

#### `logging_utils.py`
- **Purpose**: Structured logging infrastructure
- **Features**:
  - JSON-formatted logs for production
  - Pretty console output for development
  - Backwards-compatible with file-based logging
  - Automatic timestamp and log level injection
- **Key Components**:
  - `configure_logging()` - Initialize logging system
  - `get_logger()` - Get logger instance
  - `LoggerAdapter` - Bridge for legacy code

**Example Usage**:
```python
from src.utils.logging_utils import get_logger, configure_logging

configure_logging(debug=True)
logger = get_logger("my_module")
logger.info("operation_completed", user_id=123, duration_ms=45)
```

#### `redis_utils.py`
- **Purpose**: Redis connection management
- **Features**:
  - Automatic connection initialization
  - Support for Redis URL or host/port config
  - SSL support
  - Connection health checks
  - Robust error handling
- **Key Functions**:
  - `init_redis_client()` - Initialize Redis client
  - `test_redis_connection()` - Verify connectivity

**Example Usage**:
```python
from src.utils.redis_utils import init_redis_client, test_redis_connection

redis_client = init_redis_client()
if test_redis_connection(redis_client):
    print("Redis connected!")
```

#### `constants.py`
- **Purpose**: Centralized configuration and constants
- **Benefits**:
  - Eliminates magic numbers
  - Single source of truth for configuration
  - Easy to modify and test
  - Type-safe constant access
- **Categories**:
  - Redis configuration
  - Message processing settings
  - Calendar/business hours
  - OpenAI model configuration
  - Pricing formula constants
  - Rate limiting values
  - Retry/circuit breaker settings

**Example Usage**:
```python
from src.utils.constants import (
    MAX_HISTORY_LENGTH,
    BUSINESS_HOURS_START,
    PRICING_MIN_PRICE
)
```

---

## Core Modules

### `validation.py`
Input validation and sanitization for all user inputs.

**Validates**:
- Phone numbers (Greek format)
- Dates and times
- Customer names
- Event IDs
- Durations and prices
- Text inputs

**Example**:
```python
from validation import validate_phone_number, ValidationError

try:
    phone = validate_phone_number("+30 691 234 5678")
    # Returns: "6912345678"
except ValidationError as e:
    print(f"Invalid phone: {e}")
```

### `calendar_functions.py`
Google Calendar integration with booking management.

**Features**:
- Check availability with smart scheduling
- Create/cancel/reschedule bookings
- Input validation on all operations
- Temporary slot holding to prevent double-bookings
- Duration calculations from pricing

---

## Design Patterns Used

### 1. **Circuit Breaker Pattern**
- **Where**: `openai_service.py`
- **Why**: Prevents cascading failures when OpenAI API is down
- **Implementation**: Using `tenacity` library with exponential backoff
- **Configuration**: 3 retry attempts, 2-16 second wait times

### 2. **Service Layer Pattern**
- **Where**: `src/services/`
- **Why**: Separates business logic from routes and data access
- **Benefits**: Easier testing, reusability, maintainability

### 3. **Dependency Injection**
- **Where**: Service constructors
- **Why**: Allows easy mocking in tests
- **Example**: Services accept API keys as parameters with env var defaults

### 4. **Strategy Pattern**
- **Where**: Message splitting in `messaging_service.py`
- **Why**: Different strategies for splitting long messages (newline, space, hard cutoff)

---

## Configuration Management

### Environment Variables

All sensitive configuration is managed through environment variables:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_DEFAULT=gpt-4o
OPENAI_MODEL_VISION=gpt-4o-mini

# Instagram
IG_USER_ACCESS_TOKEN=...

# Redis
REDIS_URL=redis://...
# OR
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_SSL=false

# Google Calendar
GOOGLE_TOKEN_FILE=token.json
GOOGLE_CLIENT_SECRETS=credentials.json

# Application
DEBUG=false
APP_LOG_FILE=app.log
```

### Constants vs Environment Variables

- **Constants** (`src/utils/constants.py`): Application defaults and business logic values
- **Environment Variables**: Deployment-specific configuration and secrets

---

## Error Handling Strategy

### Layers of Error Handling

1. **Input Validation** - Catch bad inputs before processing
2. **Service Layer** - Retry logic for transient failures
3. **API Layer** - HTTP error responses with appropriate status codes
4. **Logging** - Structured logs for debugging and monitoring

### Example Error Flow

```
User Input → Validation → Service (with retries) → External API
    ↓           ↓              ↓                      ↓
  ValidationError  →  Log & Return Error Response
                ServiceError  →  Retry (up to 3x)
                                APIError  →  Log & Fallback
```

---

## Logging Strategy

### Structured Logging Format

All logs use JSON format in production:

```json
{
  "event": "openai_chat_completion_request",
  "timestamp": "2024-11-15T10:30:45.123Z",
  "level": "info",
  "model": "gpt-4o",
  "message_count": 5,
  "has_tools": true
}
```

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication that something unexpected happened
- **ERROR**: Serious problem, function could not complete
- **CRITICAL**: Very serious error, program may not be able to continue

### Key Events Logged

- API requests/responses
- User message processing
- Redis operations
- Calendar bookings
- Image downloads and analysis
- Error conditions

---

## Testing Strategy

### Test Structure

```
tests/
├── test_validation.py       # 40+ validation tests
├── test_calendar_functions.py  # Calendar utility tests
└── conftest.py              # Shared fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v

# Run specific test
pytest tests/test_validation.py::TestPhoneNumberValidation::test_valid_greek_mobile
```

### Test Coverage Goals

- **Validation**: 100% coverage ✅
- **Services**: 80% coverage (target)
- **Calendar Functions**: 70% coverage (target)
- **Overall**: 60%+ coverage

---

## Performance Considerations

### Caching Strategy

- **Redis**: Conversation context (7-day TTL), message queues (10-min TTL)
- **Pinecone**: Similarity search results for consistent responses

### Rate Limiting

- **Webhook**: 100 requests/hour
- **General**: 50 requests/hour, 200 requests/day
- **Storage**: Redis-backed for distributed rate limiting

### Optimization Opportunities

1. **Connection Pooling**: Add `requests.Session()` for HTTP requests
2. **Embedding Caching**: Cache sentence transformer embeddings
3. **Task Queue**: Replace `threading.Timer` with Celery/RQ
4. **Database**: Consider PostgreSQL for persistent storage

---

## Security Best Practices

### Implemented

✅ Input validation on all user inputs
✅ Rate limiting on webhooks
✅ Dependency version pinning
✅ Environment variable for secrets
✅ SQL injection prevention (parameterized queries)
✅ XSS prevention (input sanitization)

### Recommended

⚠️ Add CSRF protection for future web forms
⚠️ Implement request signing for webhooks
⚠️ Add API key rotation mechanism
⚠️ Enable audit logging for sensitive operations

---

## Migration Guide

### For Existing Deployments

To migrate from the old monolithic structure to the new modular architecture:

1. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   - No changes needed - all existing env vars still work

3. **Code Changes**:
   - Old code still works (backwards compatible)
   - New services can be adopted gradually

4. **Testing**:
   ```bash
   pytest  # Run test suite to verify everything works
   ```

### Gradual Adoption

You can adopt the new modules incrementally:

```python
# Old way (still works)
from calendar_functions import create_booking

# New way (recommended)
from src.services.openai_service import OpenAIService
service = OpenAIService()
```

---

## Future Enhancements

### Planned

1. **Complete Refactoring**: Extract remaining code from `app.py`
   - Intent service
   - Message queue service
   - Routes layer

2. **Database Layer**: Add PostgreSQL for:
   - Persistent conversation history
   - Analytics and metrics
   - Audit logs

3. **Monitoring**: Add:
   - Prometheus metrics
   - Sentry error tracking
   - Health check dashboard

4. **Deployment**: Containerization
   - Docker multi-stage builds
   - Kubernetes deployment manifests
   - CI/CD with GitHub Actions

---

## Troubleshooting

### Common Issues

#### "OpenAI API Key Missing"
```
ValueError: OPENAI_API_KEY environment variable is required
```
**Solution**: Set `OPENAI_API_KEY` in your `.env` file

#### "Redis Connection Failed"
```
redis.exceptions.ConnectionError: Error connecting to Redis
```
**Solution**: Check `REDIS_URL` or `REDIS_HOST`/`REDIS_PORT` configuration

#### "Import Error: No module named 'src'"
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from project root directory or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

---

## Contributing

### Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions
- Keep functions under 50 lines
- Maximum line length: 100 characters

### Adding New Services

1. Create service in `src/services/`
2. Add structured logging
3. Implement error handling with retries
4. Write unit tests
5. Update this documentation

### Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No secrets in code
- [ ] Logging added for key operations

---

## Contact & Support

For questions about this architecture:
- Review this document
- Check existing code examples
- Read inline code documentation

---

*Last Updated: 2024-11-15*
*Architecture Version: 2.0*
