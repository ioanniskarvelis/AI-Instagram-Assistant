"""
Unit tests for calendar functions
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from calendar_functions import (
    round_duration_to_5_minutes,
    format_duration_display,
    format_available_slots_message
)


class TestRoundDuration:
    """Tests for duration rounding function"""

    def test_round_exact_5_minutes(self):
        """Test rounding when duration is already a multiple of 5 minutes"""
        # 1 hour = 60 minutes, already a multiple of 5
        result = round_duration_to_5_minutes(1.0)
        assert result == 1.0

    def test_round_up_to_5_minutes(self):
        """Test rounding up to nearest 5 minutes"""
        # 1.08 hours = 65 minutes, should round to 65 minutes (1.0833 hours)
        result = round_duration_to_5_minutes(1.08)
        assert result == pytest.approx(1.0833, rel=0.01)

    def test_round_small_duration(self):
        """Test rounding small duration"""
        # 0.5 hours = 30 minutes, already a multiple of 5
        result = round_duration_to_5_minutes(0.5)
        assert result == 0.5

    def test_round_fractional_minutes(self):
        """Test rounding fractional minutes"""
        # 1.12 hours = 67.2 minutes, should round up to 70 minutes
        result = round_duration_to_5_minutes(1.12)
        assert result == pytest.approx(1.1667, rel=0.01)


class TestFormatDuration:
    """Tests for duration formatting function"""

    def test_format_one_hour(self):
        """Test formatting 1 hour"""
        result = format_duration_display(1.0)
        assert result == "1 ώρα"

    def test_format_multiple_hours(self):
        """Test formatting multiple hours"""
        result = format_duration_display(2.0)
        assert result == "2 ώρες"

    def test_format_minutes_only(self):
        """Test formatting minutes only"""
        result = format_duration_display(0.5)  # 30 minutes
        assert result == "30 λεπτά"

    def test_format_hours_and_minutes(self):
        """Test formatting hours and minutes"""
        result = format_duration_display(1.5)  # 1 hour 30 minutes
        assert result == "1 ώρα και 30 λεπτά"

    def test_format_multiple_hours_and_minutes(self):
        """Test formatting multiple hours and minutes"""
        result = format_duration_display(2.75)  # 2 hours 45 minutes
        assert result == "2 ώρες και 45 λεπτά"


class TestFormatAvailableSlots:
    """Tests for available slots message formatting"""

    def test_empty_slots(self):
        """Test formatting empty slots list"""
        result = format_available_slots_message([])
        assert "Δυστυχώς δεν υπάρχουν" in result

    def test_single_slot(self):
        """Test formatting single slot"""
        slots = [
            {
                'date': '2024-11-20',
                'start_time': '14:00',
                'datetime': '2024-11-20T14:00:00+02:00'
            }
        ]
        result = format_available_slots_message(slots)
        assert "Διαθέσιμες ώρες" in result
        assert "14:00" in result

    def test_multiple_slots_same_day(self):
        """Test formatting multiple slots on same day"""
        slots = [
            {
                'date': '2024-11-20',
                'start_time': '14:00',
                'datetime': '2024-11-20T14:00:00+02:00'
            },
            {
                'date': '2024-11-20',
                'start_time': '15:00',
                'datetime': '2024-11-20T15:00:00+02:00'
            },
            {
                'date': '2024-11-20',
                'start_time': '16:00',
                'datetime': '2024-11-20T16:00:00+02:00'
            }
        ]
        result = format_available_slots_message(slots)
        assert "14:00" in result
        assert "15:00" in result
        assert "16:00" in result

    def test_slots_different_days(self):
        """Test formatting slots across different days"""
        slots = [
            {
                'date': '2024-11-20',
                'start_time': '14:00',
                'datetime': '2024-11-20T14:00:00+02:00'
            },
            {
                'date': '2024-11-21',
                'start_time': '15:00',
                'datetime': '2024-11-21T15:00:00+02:00'
            }
        ]
        result = format_available_slots_message(slots)
        # Should contain dates formatted in Greek
        assert "Διαθέσιμες ώρες" in result


class TestCalendarIntegration:
    """Integration tests for calendar functions (mocked)"""

    @patch('calendar_functions.service')
    def test_create_booking_validates_inputs(self, mock_service):
        """Test that create_booking validates inputs"""
        from calendar_functions import create_booking

        # This should fail validation due to invalid phone
        result = create_booking(
            mock_service,
            "John Doe",
            "invalid",  # Invalid phone
            "2024-11-20",
            "14:00"
        )
        assert result is None  # Should return None on validation error

    @patch('calendar_functions.service')
    def test_cancel_booking_validates_event_id(self, mock_service):
        """Test that cancel_booking validates event ID"""
        from calendar_functions import cancel_booking

        # This should fail validation due to invalid event ID
        result = cancel_booking(
            mock_service,
            "invalid-event-id!"  # Invalid characters
        )
        assert result is False  # Should return False on validation error
