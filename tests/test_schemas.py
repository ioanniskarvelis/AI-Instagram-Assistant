import pytest
from pydantic import ValidationError


def _payload(message: dict) -> dict:
    return {
        "object": "instagram",
        "entry": [
            {
                "id": "STUDIO",
                "time": 1723000000,
                "messaging": [
                    {
                        "sender": {"id": "SENDER_1"},
                        "recipient": {"id": "STUDIO"},
                        "timestamp": 1723000000,
                        "message": message,
                    }
                ],
            }
        ],
    }


def test_text_message_is_replyable():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m1", "text": "Γεια σας, πόσο κοστίζει;"})
    )
    assert payload.replyable_messages("STUDIO") == [
        ("SENDER_1", "Γεια σας, πόσο κοστίζει;")
    ]


def test_echo_message_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m2", "text": "our own reply", "is_echo": True})
    )
    assert payload.replyable_messages("STUDIO") == []


def test_message_without_text_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m3", "attachments": [{"type": "image"}]})
    )
    assert payload.replyable_messages("STUDIO") == []


def test_event_without_message_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        {
            "object": "instagram",
            "entry": [
                {
                    "id": "STUDIO",
                    "messaging": [
                        {
                            "sender": {"id": "SENDER_1"},
                            "recipient": {"id": "STUDIO"},
                            "read": {"mid": "m4"},
                        }
                    ],
                }
            ],
        }
    )
    assert payload.replyable_messages("STUDIO") == []


def test_multiple_entries_and_events_are_all_collected():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        {
            "object": "instagram",
            "entry": [
                {
                    "id": "STUDIO",
                    "messaging": [
                        {
                            "sender": {"id": "A"},
                            "recipient": {"id": "STUDIO"},
                            "message": {"mid": "m1", "text": "one"},
                        },
                        {
                            "sender": {"id": "B"},
                            "recipient": {"id": "STUDIO"},
                            "message": {"mid": "m2", "text": "two"},
                        },
                    ],
                },
                {
                    "id": "STUDIO",
                    "messaging": [
                        {
                            "sender": {"id": "C"},
                            "recipient": {"id": "STUDIO"},
                            "message": {"mid": "m3", "text": "three"},
                        }
                    ],
                },
            ],
        }
    )
    assert payload.replyable_messages("STUDIO") == [
        ("A", "one"),
        ("B", "two"),
        ("C", "three"),
    ]


def test_entry_for_a_different_account_is_ignored():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        {
            "object": "instagram",
            "entry": [
                {
                    "id": "OTHER_ACCOUNT",
                    "messaging": [
                        {
                            "sender": {"id": "A"},
                            "recipient": {"id": "OTHER_ACCOUNT"},
                            "message": {"mid": "m1", "text": "one"},
                        }
                    ],
                }
            ],
        }
    )
    assert payload.replyable_messages("STUDIO") == []


def test_payload_missing_object_is_rejected():
    from app.schemas import WebhookPayload

    with pytest.raises(ValidationError):
        WebhookPayload.model_validate({"entry": []})
