"""
Pytest configuration and shared fixtures
"""

import pytest
from datetime import datetime, timedelta


@pytest.fixture
def valid_phone_number():
    """Fixture for a valid Greek phone number"""
    return "6912345678"


@pytest.fixture
def future_date():
    """Fixture for a valid future date"""
    return (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')


@pytest.fixture
def valid_time():
    """Fixture for a valid business hours time"""
    return "14:00"


@pytest.fixture
def mock_calendar_service(mocker):
    """Fixture for mocked Google Calendar service"""
    return mocker.Mock()


@pytest.fixture
def sample_calendar_slots():
    """Fixture for sample available calendar slots"""
    return [
        {
            'date': '2024-11-20',
            'start_time': '11:00',
            'datetime': '2024-11-20T11:00:00+02:00'
        },
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
