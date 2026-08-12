import logging
from enum import StrEnum

import anthropic

from app.config import get_settings
from app.history import Turn

logger = logging.getLogger(__name__)


class Intent(StrEnum):
    """One of four reply-phrasing strategies a customer message maps to."""

    PRICE = "price"
    BOOKING = "booking"
    DESIGN = "design"
    GENERAL = "general"


CLASSIFY_SYSTEM_PROMPT = """You classify Instagram DM messages sent to a tattoo \
studio into exactly one of four intents, based on the customer's most recent \
message and the conversation so far.

- price: asking about cost, price, deposit, or rate. E.g. "Πόσο κοστίζει ένα \
μικρό τατουάζ;", "How much would this cost?", "τι τιμή έχει;"
- booking: asking about availability, wanting to schedule, reschedule, or \
cancel an appointment. E.g. "Έχετε ραντεβού αυτή την εβδομάδα;", "Can I book \
for next Friday?", "θέλω να ακυρώσω το ραντεβού μου"
- design: describing or discussing a tattoo idea, placement, size, style, or \
reference images. E.g. "Θέλω ένα μικρό τριαντάφυλλο στον καρπό", "I'm thinking \
of a fine-line piece on my forearm"
- general: anything else — greetings, thanks, small talk, studio-info \
questions (hours, location, artist), or unclear messages.

Call the classify_intent tool with exactly one of these four values."""

_TOOL_NAME = "classify_intent"
_TOOL = {
    "name": _TOOL_NAME,
    "description": "Record the classified intent of the customer's message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [i.value for i in Intent],
            }
        },
        "required": ["intent"],
    },
}

_client = anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)


async def classify(turns: list[Turn]) -> Intent:
    """Classify the customer's intent from the recent conversation window.

    Returns Intent.GENERAL on any failure — API error, refusal, a response
    with no tool_use block, or an unrecognized intent value. Never raises:
    a classification problem must never block a reply from being generated,
    the same degrade-silently posture as app.rag.retrieve returning [].
    """
    try:
        settings = get_settings()
        response = await _client.messages.create(
            model=settings.intent_model,
            max_tokens=settings.intent_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": CLASSIFY_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": turn.role, "content": turn.text} for turn in turns],
        )
    except anthropic.APIError:
        logger.exception("Anthropic call failed during intent classification")
        return Intent.GENERAL
    except Exception:
        logger.exception("Unexpected error classifying intent")
        return Intent.GENERAL

    if response.stop_reason == "refusal":
        logger.warning("Model declined to classify (stop_reason=refusal)")
        return Intent.GENERAL

    for block in response.content:
        if block.type == "tool_use" and block.name == _TOOL_NAME:
            value = block.input.get("intent")
            try:
                return Intent(value)
            except ValueError:
                logger.warning("Unrecognized intent value from classifier: %r", value)
                return Intent.GENERAL

    logger.warning(
        "No tool_use block in classification response (stop_reason=%s)",
        response.stop_reason,
    )
    return Intent.GENERAL
