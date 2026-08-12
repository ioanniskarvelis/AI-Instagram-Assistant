from pydantic import BaseModel, ConfigDict, Field


class Participant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str


class AttachmentPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None


class Attachment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    payload: AttachmentPayload | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mid: str | None = None
    text: str | None = None
    is_echo: bool = False
    attachments: list[Attachment] = Field(default_factory=list)


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
        """Return (sender_id, text) for inbound events worth replying to.

        An attachment (image, video, audio, file, ...) with no caption text
        becomes a bracketed placeholder rather than being dropped — otherwise
        a burst like "tattoo" / [image] / "on my arm" would silently lose the
        image and the assistant would never know one arrived. The placeholder
        flows through history and debounce buffering exactly like real text;
        the model reasons about the fact an image was sent, not its contents
        (no vision support here).

        Skips echoes — the studio's own outbound replies are delivered back to
        this webhook, and replying to them makes the assistant answer itself in
        a loop. Also skips events with neither text nor attachments: reactions,
        read receipts and delivery receipts all arrive on this same endpoint.
        Also skips entries for any Instagram account other than `account_id` —
        the same app/webhook can be subscribed to more than one account.
        """
        replyable: list[tuple[str, str]] = []
        for entry in self.entry:
            if entry.id != account_id:
                continue
            for event in entry.messaging:
                message = event.message
                if message is None or message.is_echo:
                    continue
                if message.text:
                    replyable.append((event.sender.id, message.text))
                elif message.attachments:
                    placeholder = " ".join(
                        _attachment_placeholder(a) for a in message.attachments
                    )
                    replyable.append((event.sender.id, placeholder))
        return replyable

    def image_urls(self, account_id: str) -> list[tuple[str, str]]:
        """Return (sender_id, url) for every image attachment across inbound,
        non-echo messages for `account_id`.

        Separate from replyable_messages(), which only ever reports a
        bracketed placeholder for an attachment — this is what lets the
        quote-request flow (app.webhook / app.quotes) actually hand the
        artists real reference photos instead of just knowing one exists.
        """
        urls: list[tuple[str, str]] = []
        for entry in self.entry:
            if entry.id != account_id:
                continue
            for event in entry.messaging:
                message = event.message
                if message is None or message.is_echo:
                    continue
                for attachment in message.attachments:
                    if attachment.type == "image" and attachment.payload and attachment.payload.url:
                        urls.append((event.sender.id, attachment.payload.url))
        return urls


def _attachment_placeholder(attachment: Attachment) -> str:
    if attachment.type == "image":
        return "[the customer sent an image]"
    return f"[the customer sent a {attachment.type or 'file'} attachment]"
