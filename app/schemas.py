from pydantic import BaseModel, ConfigDict, Field


class Participant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mid: str | None = None
    text: str | None = None
    is_echo: bool = False


class MessagingEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sender: Participant
    recipient: Participant
    message: Message | None = None


class Entry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    messaging: list[MessagingEvent] = Field(default_factory=list)


class WebhookPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    object: str
    entry: list[Entry] = Field(default_factory=list)

    def replyable_messages(self, account_id: str) -> list[tuple[str, str]]:
        """Return (sender_id, text) for inbound text messages worth replying to.

        Skips echoes — the studio's own outbound replies are delivered back to
        this webhook, and replying to them makes the assistant answer itself in
        a loop. Also skips events with no text: reactions, attachments, read
        receipts and delivery receipts all arrive on this same endpoint.
        Also skips entries for any Instagram account other than `account_id` —
        the same app/webhook can be subscribed to more than one account.
        """
        replyable: list[tuple[str, str]] = []
        for entry in self.entry:
            if entry.id != account_id:
                continue
            for event in entry.messaging:
                message = event.message
                if message is None or message.is_echo or not message.text:
                    continue
                replyable.append((event.sender.id, message.text))
        return replyable
