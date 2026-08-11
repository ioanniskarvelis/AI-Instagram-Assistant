import pytest
from pydantic import ValidationError


def _payload(message: dict) -> dict:
    return {
        "object": "instagram",
        "entry": [
            {
                "id": "17841400000000000",
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
    assert payload.replyable_messages() == [
        ("SENDER_1", "Γεια σας, πόσο κοστίζει;")
    ]


def test_echo_message_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m2", "text": "our own reply", "is_echo": True})
    )
    assert payload.replyable_messages() == []


def test_message_without_text_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        _payload({"mid": "m3", "attachments": [{"type": "image"}]})
    )
    assert payload.replyable_messages() == []


def test_event_without_message_is_skipped():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        {
            "object": "instagram",
            "entry": [
                {
                    "id": "E1",
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
    assert payload.replyable_messages() == []


def test_multiple_entries_and_events_are_all_collected():
    from app.schemas import WebhookPayload

    payload = WebhookPayload.model_validate(
        {
            "object": "instagram",
            "entry": [
                {
                    "id": "E1",
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
                    "id": "E2",
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
    assert payload.replyable_messages() == [
        ("A", "one"),
        ("B", "two"),
        ("C", "three"),
    ]


def test_payload_missing_object_is_rejected():
    from app.schemas import WebhookPayload

    with pytest.raises(ValidationError):
        WebhookPayload.model_validate({"entry": []})
