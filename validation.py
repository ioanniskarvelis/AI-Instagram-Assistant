"""
Input validation utilities for the Instagram Assistant
Provides validation functions for phone numbers, dates, times, and other inputs
"""

import re
from datetime import datetime
from typing import Optional, Tuple


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_phone_number(phone: str) -> str:
    """
    Validate and sanitize Greek phone numbers

    Args:
        phone: Phone number string (can include +30, spaces, etc.)

    Returns:
        Sanitized 10-digit phone number

    Raises:
        ValidationError: If phone number is invalid
    """
    if not phone:
        raise ValidationError("Phone number cannot be empty")

    # Remove common formatting characters
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)

    # Remove country code if present
    if cleaned.startswith('+30'):
        cleaned = cleaned[3:]
    elif cleaned.startswith('0030'):
        cleaned = cleaned[4:]
    elif cleaned.startswith('30') and len(cleaned) == 12:
        cleaned = cleaned[2:]

    # Validate length
    if len(cleaned) != 10:
        raise ValidationError(f"Invalid phone number length: {len(cleaned)} (expected 10 digits)")

    # Validate all digits
    if not cleaned.isdigit():
        raise ValidationError("Phone number must contain only digits")

    # Validate Greek mobile number patterns (69, 68) or landline (21, 22, 23, 24, 25, 26, 27, 28, 29)
    if not (cleaned.startswith('69') or cleaned.startswith('68') or
            (cleaned.startswith('2') and len(cleaned) == 10)):
        raise ValidationError(f"Invalid Greek phone number pattern: {cleaned[:2]}")

    return cleaned


def validate_date(date_str: str, date_format: str = '%Y-%m-%d') -> datetime:
    """
    Validate and parse date string

    Args:
        date_str: Date string to validate
        date_format: Expected date format (default: YYYY-MM-DD)

    Returns:
        Parsed datetime object

    Raises:
        ValidationError: If date is invalid or in the past
    """
    if not date_str:
        raise ValidationError("Date cannot be empty")

    try:
        parsed_date = datetime.strptime(date_str, date_format)
    except ValueError as e:
        raise ValidationError(f"Invalid date format: {date_str} (expected {date_format})")

    # Check if date is not in the past (allow today)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if parsed_date < today:
        raise ValidationError(f"Date cannot be in the past: {date_str}")

    return parsed_date


def validate_time(time_str: str, time_format: str = '%H:%M') -> Tuple[int, int]:
    """
    Validate and parse time string

    Args:
        time_str: Time string to validate (e.g., "14:30")
        time_format: Expected time format (default: HH:MM)

    Returns:
        Tuple of (hours, minutes)

    Raises:
        ValidationError: If time is invalid
    """
    if not time_str:
        raise ValidationError("Time cannot be empty")

    try:
        parsed_time = datetime.strptime(time_str, time_format)
    except ValueError:
        raise ValidationError(f"Invalid time format: {time_str} (expected {time_format})")

    hours = parsed_time.hour
    minutes = parsed_time.minute

    # Validate business hours (11:00 - 20:00)
    if hours < 11 or hours >= 20:
        raise ValidationError(f"Time must be within business hours (11:00-20:00): {time_str}")

    return hours, minutes


def validate_duration(duration_hours: Optional[float], tattoo_price: Optional[float] = None) -> float:
    """
    Validate appointment duration

    Args:
        duration_hours: Duration in hours (can be None if tattoo_price provided)
        tattoo_price: Price in euros (used to calculate duration if duration_hours not provided)

    Returns:
        Validated duration in hours

    Raises:
        ValidationError: If duration is invalid
    """
    if duration_hours is None and tattoo_price is None:
        return 1.0  # Default to 1 hour

    if duration_hours is not None:
        if duration_hours <= 0:
            raise ValidationError(f"Duration must be positive: {duration_hours}")
        if duration_hours > 10:
            raise ValidationError(f"Duration too long (max 10 hours): {duration_hours}")
        return duration_hours

    # Calculate from price
    if tattoo_price <= 0:
        raise ValidationError(f"Price must be positive: {tattoo_price}")
    if tattoo_price > 5000:
        raise ValidationError(f"Price too high (max 5000€): {tattoo_price}")

    # Use the same formula as in the app: price / 100 = hours
    calculated_duration = tattoo_price / 100

    # Round to nearest 5 minutes
    import math
    total_minutes = calculated_duration * 60
    rounded_minutes = math.ceil(total_minutes / 5) * 5

    return rounded_minutes / 60


def validate_customer_name(name: str) -> str:
    """
    Validate customer name

    Args:
        name: Customer name

    Returns:
        Stripped name

    Raises:
        ValidationError: If name is invalid
    """
    if not name:
        raise ValidationError("Customer name cannot be empty")

    name = name.strip()

    if len(name) < 2:
        raise ValidationError(f"Customer name too short: {name}")

    if len(name) > 100:
        raise ValidationError(f"Customer name too long (max 100 chars): {name[:20]}...")

    # Allow only letters, spaces, and common Greek/Latin characters
    if not re.match(r'^[a-zA-ZΑ-Ωα-ωίϊΐόάέύϋΰήώ\s\-\.]+$', name):
        raise ValidationError(f"Customer name contains invalid characters: {name}")

    return name


def validate_event_id(event_id: str) -> str:
    """
    Validate Google Calendar event ID

    Args:
        event_id: Event ID string

    Returns:
        Validated event ID

    Raises:
        ValidationError: If event ID is invalid
    """
    if not event_id:
        raise ValidationError("Event ID cannot be empty")

    # Google Calendar event IDs are alphanumeric with underscores
    if not re.match(r'^[a-zA-Z0-9_]+$', event_id):
        raise ValidationError(f"Invalid event ID format: {event_id}")

    if len(event_id) > 1024:
        raise ValidationError("Event ID too long")

    return event_id


def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize general text input (descriptions, etc.)

    Args:
        text: Text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        ValidationError: If text is invalid
    """
    if not text:
        return ""

    text = text.strip()

    if len(text) > max_length:
        raise ValidationError(f"Text too long (max {max_length} chars): {len(text)} chars")

    # Remove any potential control characters but keep Greek and common punctuation
    # Allow: letters (Greek/Latin), digits, spaces, and common punctuation
    sanitized = re.sub(r'[^\w\s\.,!?;:\-\(\)\/\"\'\n]', '', text, flags=re.UNICODE)

    return sanitized
