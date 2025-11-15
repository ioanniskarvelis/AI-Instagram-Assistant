"""
Application constants and configuration values
Centralizes magic numbers and configuration for better maintainability
"""

# Redis Configuration
MAX_HISTORY_LENGTH = 20  # Maximum number of conversation messages to keep
GRACE_WINDOW_SECONDS = 20  # Seconds to wait before processing batched messages
HOLD_TTL_SECONDS = 30 * 60  # 30 minutes - how long to hold calendar slots
CONVERSATION_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days - conversation context expiry
QUEUE_TTL_SECONDS = 60 * 10  # 10 minutes - message queue expiry
MUTE_DURATION_SECONDS = 60 * 60 * 2  # 2 hours - human override mute duration

# Message Processing
MESSAGE_MAX_LENGTH = 800  # Maximum Instagram message length before splitting
PROCESSING_LOCK_TTL = 30  # Seconds before processing lock expires

# Calendar Configuration
BUSINESS_HOURS_START = 11  # Opening hour (24h format)
BUSINESS_HOURS_END = 20  # Closing hour (24h format)
WORKING_DAYS = [0, 1, 2, 3, 4, 5]  # Monday=0 to Saturday=5 (Sunday=6 is closed)

# OpenAI Configuration
OPENAI_MODEL_DEFAULT = "gpt-4o"
OPENAI_MODEL_VISION = "gpt-4o-mini"
OPENAI_MODEL_CLASSIFY = "gpt-4o-mini"
OPENAI_TEMPERATURE_DEFAULT = 1.0
OPENAI_TEMPERATURE_PRICING = 0.3
OPENAI_TEMPERATURE_CLASSIFY = 0.0

# Pricing Formula Constants
PRICING_MIN_PRICE = 45  # Minimum price in euros
PRICING_ROUNDING = 5  # Round prices to nearest 5 euros
PRICING_FORMULA_DIVISOR = 5  # Used in pricing calculation
PRICING_INK_MULTIPLIER = 0.3  # Ink density multiplier
PRICING_DURATION_FORMULA = 100  # Price to hours conversion: price/100 = hours
MAX_TATTOO_PRICE = 5000  # Maximum price validation limit
MAX_DURATION_HOURS = 10  # Maximum appointment duration in hours

# Multi-tattoo Pricing
MULTI_TATTOO_DISCOUNT = 0.10  # 10% discount for multiple tattoos

# Pinecone Configuration
CONVERSATIONS_INDEX_NAME = "tattoo-conversations"
PRICING_INDEX_NAME = "tattoo-pricing"
SIMILARITY_THRESHOLD = 0.75  # Minimum similarity score for RAG retrieval
TOP_K_SIMILAR = 3  # Number of similar conversations to retrieve

# Rate Limiting
RATE_LIMIT_WEBHOOK_HOURLY = 100  # Webhook requests per hour
RATE_LIMIT_GLOBAL_HOURLY = 50  # General requests per hour
RATE_LIMIT_GLOBAL_DAILY = 200  # General requests per day

# Retry Configuration (for circuit breaker)
MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts for API calls
RETRY_WAIT_EXPONENTIAL_MULTIPLIER = 2  # Exponential backoff multiplier
RETRY_WAIT_MIN_SECONDS = 2  # Minimum wait between retries
RETRY_WAIT_MAX_SECONDS = 16  # Maximum wait between retries

# Admin Configuration
# Note: These IDs should be set via environment variables in production
DEFAULT_ADMIN_SENDER_IDS = ["1018827777120359", "6815817155197102"]
REACTION_BOT_SENDER_ID_DEFAULT = "17841463333962356"

# File Paths
PROMPTS_DIR = "./prompts"
PRICING_PROMPT_FILE = f"{PROMPTS_DIR}/pricing.txt"
BOOKING_PROMPT_FILE = f"{PROMPTS_DIR}/booking.txt"
INFORMATION_PROMPT_FILE = f"{PROMPTS_DIR}/information.txt"
FOLLOWUP_PROMPT_FILE = f"{PROMPTS_DIR}/follow_up.txt"
CLASSIFICATION_PROMPT_FILE = f"{PROMPTS_DIR}/classification.txt"

# Intent Priorities (lower number = higher priority)
INTENT_PRIORITIES = {
    "pricing": 1,
    "booking_request": 2,
    "studio_information": 3,
    "follow_up": 4,
    "other": 5
}
