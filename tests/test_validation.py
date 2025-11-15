"""
Unit tests for validation module
"""

import pytest
from datetime import datetime, timedelta
from validation import (
    validate_phone_number,
    validate_date,
    validate_time,
    validate_duration,
    validate_customer_name,
    validate_event_id,
    sanitize_text_input,
    ValidationError
)


class TestPhoneNumberValidation:
    """Tests for phone number validation"""

    def test_valid_greek_mobile(self):
        """Test valid Greek mobile number"""
        assert validate_phone_number("6912345678") == "6912345678"

    def test_valid_with_country_code(self):
        """Test valid number with +30 country code"""
        assert validate_phone_number("+30 6912345678") == "6912345678"

    def test_valid_with_spaces(self):
        """Test valid number with spaces"""
        assert validate_phone_number("691 234 5678") == "6912345678"

    def test_valid_with_dashes(self):
        """Test valid number with dashes"""
        assert validate_phone_number("691-234-5678") == "6912345678"

    def test_valid_landline(self):
        """Test valid Athens landline"""
        assert validate_phone_number("2101234567") == "2101234567"

    def test_invalid_length(self):
        """Test invalid length"""
        with pytest.raises(ValidationError, match="Invalid phone number length"):
            validate_phone_number("691234567")  # 9 digits

    def test_invalid_characters(self):
        """Test invalid characters"""
        with pytest.raises(ValidationError, match="must contain only digits"):
            validate_phone_number("69123abc78")

    def test_invalid_pattern(self):
        """Test invalid Greek number pattern"""
        with pytest.raises(ValidationError, match="Invalid Greek phone number pattern"):
            validate_phone_number("5912345678")  # Doesn't start with valid prefix

    def test_empty_phone(self):
        """Test empty phone number"""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_phone_number("")


class TestDateValidation:
    """Tests for date validation"""

    def test_valid_future_date(self):
        """Test valid future date"""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        result = validate_date(tomorrow)
        assert isinstance(result, datetime)

    def test_today_is_valid(self):
        """Test that today's date is valid"""
        today = datetime.now().strftime('%Y-%m-%d')
        result = validate_date(today)
        assert isinstance(result, datetime)

    def test_past_date_invalid(self):
        """Test that past dates are invalid"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        with pytest.raises(ValidationError, match="cannot be in the past"):
            validate_date(yesterday)

    def test_invalid_format(self):
        """Test invalid date format"""
        with pytest.raises(ValidationError, match="Invalid date format"):
            validate_date("15/11/2024")  # Wrong format

    def test_empty_date(self):
        """Test empty date"""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_date("")


class TestTimeValidation:
    """Tests for time validation"""

    def test_valid_business_hours(self):
        """Test valid time within business hours"""
        hours, minutes = validate_time("14:30")
        assert hours == 14
        assert minutes == 30

    def test_opening_time(self):
        """Test opening time (11:00)"""
        hours, minutes = validate_time("11:00")
        assert hours == 11
        assert minutes == 0

    def test_before_opening_invalid(self):
        """Test time before opening hours"""
        with pytest.raises(ValidationError, match="must be within business hours"):
            validate_time("10:00")

    def test_after_closing_invalid(self):
        """Test time after closing hours"""
        with pytest.raises(ValidationError, match="must be within business hours"):
            validate_time("20:30")

    def test_invalid_format(self):
        """Test invalid time format"""
        with pytest.raises(ValidationError, match="Invalid time format"):
            validate_time("2:30 PM")

    def test_empty_time(self):
        """Test empty time"""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_time("")


class TestDurationValidation:
    """Tests for duration validation"""

    def test_valid_duration_hours(self):
        """Test valid duration in hours"""
        result = validate_duration(2.5)
        assert result == 2.5

    def test_duration_from_price(self):
        """Test duration calculated from price"""
        result = validate_duration(None, 200)  # 200€ / 100 = 2 hours
        assert result == 2.0

    def test_default_duration(self):
        """Test default duration when both are None"""
        result = validate_duration(None, None)
        assert result == 1.0

    def test_negative_duration_invalid(self):
        """Test negative duration is invalid"""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_duration(-1)

    def test_too_long_duration_invalid(self):
        """Test duration over 10 hours is invalid"""
        with pytest.raises(ValidationError, match="too long"):
            validate_duration(12)

    def test_negative_price_invalid(self):
        """Test negative price is invalid"""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_duration(None, -50)

    def test_too_high_price_invalid(self):
        """Test price over 5000€ is invalid"""
        with pytest.raises(ValidationError, match="too high"):
            validate_duration(None, 6000)


class TestCustomerNameValidation:
    """Tests for customer name validation"""

    def test_valid_greek_name(self):
        """Test valid Greek name"""
        result = validate_customer_name("Γιάννης Παπαδόπουλος")
        assert result == "Γιάννης Παπαδόπουλος"

    def test_valid_latin_name(self):
        """Test valid Latin name"""
        result = validate_customer_name("John Smith")
        assert result == "John Smith"

    def test_name_with_hyphen(self):
        """Test name with hyphen"""
        result = validate_customer_name("Mary-Jane Watson")
        assert result == "Mary-Jane Watson"

    def test_strip_whitespace(self):
        """Test that leading/trailing whitespace is stripped"""
        result = validate_customer_name("  John Doe  ")
        assert result == "John Doe"

    def test_empty_name_invalid(self):
        """Test empty name is invalid"""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_customer_name("")

    def test_too_short_name_invalid(self):
        """Test name too short"""
        with pytest.raises(ValidationError, match="too short"):
            validate_customer_name("A")

    def test_too_long_name_invalid(self):
        """Test name too long"""
        long_name = "A" * 101
        with pytest.raises(ValidationError, match="too long"):
            validate_customer_name(long_name)

    def test_invalid_characters(self):
        """Test name with invalid characters"""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_customer_name("John@Doe")


class TestEventIdValidation:
    """Tests for event ID validation"""

    def test_valid_event_id(self):
        """Test valid event ID"""
        result = validate_event_id("abc123_xyz789")
        assert result == "abc123_xyz789"

    def test_empty_event_id_invalid(self):
        """Test empty event ID is invalid"""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_event_id("")

    def test_invalid_characters(self):
        """Test event ID with invalid characters"""
        with pytest.raises(ValidationError, match="Invalid event ID format"):
            validate_event_id("event-id-123")  # Hyphens not allowed

    def test_too_long_event_id(self):
        """Test event ID that's too long"""
        long_id = "a" * 1025
        with pytest.raises(ValidationError, match="too long"):
            validate_event_id(long_id)


class TestTextSanitization:
    """Tests for text sanitization"""

    def test_valid_text(self):
        """Test valid text sanitization"""
        result = sanitize_text_input("Θέλω ένα τατουάζ στο χέρι!")
        assert "τατουάζ" in result

    def test_empty_text(self):
        """Test empty text returns empty string"""
        result = sanitize_text_input("")
        assert result == ""

    def test_strip_whitespace(self):
        """Test whitespace is stripped"""
        result = sanitize_text_input("   Hello   ")
        assert result == "Hello"

    def test_text_too_long(self):
        """Test text over max length is invalid"""
        long_text = "a" * 1001
        with pytest.raises(ValidationError, match="too long"):
            sanitize_text_input(long_text)

    def test_preserve_newlines(self):
        """Test that newlines are preserved"""
        text = "Line 1\nLine 2"
        result = sanitize_text_input(text)
        assert "\n" in result
