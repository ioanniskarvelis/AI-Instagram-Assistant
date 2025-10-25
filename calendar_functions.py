from datetime import datetime, timedelta
import pytz
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import math
import uuid
import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import redis


SCOPES = ['https://www.googleapis.com/auth/calendar']

def authenticate_google():
    """Obtain valid Google Calendar credentials.

    Logic:
    1.  If a cached token.json exists and is valid → use it.
    2.  If it is expired but refreshable → refresh it silently (no browser).
    3.  Otherwise fall back to the interactive flow **only when a TTY is attached**.

    Running inside Docker (non-interactive) with no valid token will raise a
    clear exception instead of trying to open a browser.
    """

    creds = None
    token_path = os.environ.get("GOOGLE_TOKEN_FILE", "token.json")

    # 1. Try cached credentials first
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # 2. Refresh silently if possible
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    # 3. Fallback to interactive flow only if we have a TTY (i.e., not in Docker)
    if not creds or not creds.valid:
        if os.getenv("PYTHON_ENV") == "docker":
            raise RuntimeError(
                "No valid Google credentials found.\n"
                "Run the app locally once to generate token.json or mount a pre-generated one."
            )
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        # Save for next time
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return creds

def get_calendar_service(creds):
    service = build('calendar', 'v3', credentials=creds)
    return service

# Athens timezone
ATHENS_TZ = pytz.timezone('Europe/Athens')

log_file = open(os.getenv("SCHEDULE_LOG_FILE", "schedule.log"), "a", encoding="utf-8")

# ---------------------------------------------------------------------------
# Redis setup for temporary "holds" on suggested calendar slots
# ---------------------------------------------------------------------------

def _init_redis_client():
    """Create a Redis client using REDIS_URL or host/port vars."""
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        return redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=30,
            retry_on_timeout=True,
        )
    host = os.environ.get("REDIS_HOST", "")
    if host:
        port = int(os.environ.get("REDIS_PORT", "6379"))
        username = os.environ.get("REDIS_USERNAME") or None
        password = os.environ.get("REDIS_PASSWORD") or None
        ssl = os.environ.get("REDIS_SSL", "false").lower() == "true"
        return redis.Redis(
            host=host,
            port=port,
            username=username,
            password=password,
            db=0,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=30,
            retry_on_timeout=True,
            ssl=ssl,
            ssl_cert_reqs=None if ssl else None,
        )
    return None


# Initialise the client eagerly but fail gracefully (e.g. during local testing)
try:
    _redis_client = _init_redis_client()
except Exception:
    _redis_client = None  # Redis is optional; holds simply won't work if unavailable


HOLD_TTL_SECONDS = 30 * 60  # 30 minutes


def _slot_hold_key(slot_dt):
    """Generate the Redis key for a temporary hold for a given datetime (start of slot)."""
    return f"hold:{slot_dt.strftime('%Y-%m-%dT%H:%M')}"


def round_duration_to_5_minutes(duration_hours):
    """
    Round duration up to the nearest 5-minute interval
    
    Args:
        duration_hours: Duration in hours (can be decimal)
    
    Returns:
        Duration in hours rounded up to nearest 5 minutes
    """
    # Convert to minutes
    total_minutes = duration_hours * 60
    
    # Round up to nearest 5 minutes
    rounded_minutes = math.ceil(total_minutes / 5) * 5
    
    # Convert back to hours
    return rounded_minutes / 60

def check_availability(service, start_date, end_date=None, duration_hours=None, tattoo_price=None, user_id=None, preferred_time=None):
    """
    Check available time slots in the calendar
    
    Args:
        service: Google Calendar service object
        start_date: Start date in format 'YYYY-MM-DD' or datetime object
        end_date: End date (optional, defaults to start_date)
        duration_hours: Duration of appointment in hours (if not provided, calculated from tattoo_price)
        tattoo_price: Estimated price of tattoo in euros (used to calculate duration if duration_hours not provided)
        preferred_time: Optional string in 'HH:MM' format. If provided, suggestions for the
            start_date will begin **at or after** this time instead of the opening hour. This
            allows the calling code (or the assistant) to supply the time the user actually
            asked for (e.g. "17:00") so that we avoid returning slots that are earlier in the
            same day (e.g. 11:00, 12:00, 13:00).

    Returns:
        List of available time slots (each a dict with date / start_time / datetime).
    """
    print(f"Checking availability for {start_date} to {end_date} with duration {duration_hours} and price {tattoo_price}", file=log_file)
    log_file.flush()
    # Calculate duration based on price if not explicitly provided
    if duration_hours is None:
        if tattoo_price is None:
            duration_hours = 1  # Default to 1 hour if neither is provided
        else:
            # Formula: price / 50 = number of half-hours, so price / 100 = hours
            raw_duration = tattoo_price / 100
            duration_hours = round_duration_to_5_minutes(raw_duration)
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Set working hours (e.g., 11:00 - 20:00)
    working_start = 11
    working_end = 20
    
    # Get events for the date range
    time_min = ATHENS_TZ.localize(datetime.combine(start_date, datetime.min.time()))
    time_max = ATHENS_TZ.localize(datetime.combine(end_date, datetime.max.time()))
    
    try:
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        available_slots = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip Sundays (assuming closed on Sundays)
            if current_date.weekday() == 6:
                current_date += timedelta(days=1)
                continue
            
            # Get events for this day
            day_events = []
            for event in events:
                if 'dateTime' in event['start']:
                    event_start = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
                    event_end = datetime.fromisoformat(event['end']['dateTime'].replace('Z', '+00:00'))
                    
                    # Convert to local time
                    event_start_local = event_start.astimezone(ATHENS_TZ)
                    event_end_local = event_end.astimezone(ATHENS_TZ)
                    
                    # Check if event is on current date
                    if event_start_local.date() == current_date.date():
                        day_events.append((event_start_local, event_end_local))
            
            # ------------------------------------------------------------------
            # Determine the first slot to evaluate for the current_date.
            # For the *first* requested day we respect `preferred_time` (if given).
            # For all subsequent days we start from the normal opening hour.
            # ------------------------------------------------------------------

            if preferred_time and current_date == start_date:
                try:
                    pref_dt = datetime.strptime(preferred_time, "%H:%M")
                    pref_hour = pref_dt.hour
                    pref_minute = pref_dt.minute

                    # If preferred time is before opening, use opening hour instead.
                    if pref_hour < working_start:
                        pref_hour = working_start
                        pref_minute = 0
                    # If preferred time is after closing, skip this day entirely.
                    if pref_hour >= working_end:
                        # No slots possible on this day – move to next day.
                        current_date += timedelta(days=1)
                        continue

                    slot_time = ATHENS_TZ.localize(
                        datetime.combine(
                            current_date,
                            datetime.min.time().replace(hour=pref_hour, minute=pref_minute)
                        )
                    )
                except ValueError:
                    # Fallback to opening hour if parsing fails
                    slot_time = ATHENS_TZ.localize(
                        datetime.combine(current_date, datetime.min.time().replace(hour=working_start))
                    )
            else:
                slot_time = ATHENS_TZ.localize(
                    datetime.combine(current_date, datetime.min.time().replace(hour=working_start))
                )
            
            working_end_time = ATHENS_TZ.localize(datetime.combine(current_date, datetime.min.time().replace(hour=working_end)))
            
            while slot_time + timedelta(hours=duration_hours) <= working_end_time:
                slot_end = slot_time + timedelta(hours=duration_hours)
                
                # Count overlapping events for this potential slot
                overlapping_count = 0
                for event_start, event_end in day_events:
                    # Check if the event overlaps with the potential slot
                    if not (event_end <= slot_time or event_start >= slot_end):
                        overlapping_count += 1
                
                # Only add slot if less than 2 appointments overlap and it isn't held by another user
                if overlapping_count < 2:
                    slot_key = _slot_hold_key(slot_time)
                    holder = _redis_client.get(slot_key) if _redis_client else None

                    # Skip if someone else is already holding this slot
                    if holder and holder != str(user_id):
                        pass
                    else:
                        available_slots.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'start_time': slot_time.strftime('%H:%M'),
                            'datetime': slot_time.isoformat()
                        })
 
                # Move to next hour
                slot_time += timedelta(hours=1)
            
            current_date += timedelta(days=1)
        
        # We only want to hold the slots that we will suggest (first 10)
        suggested_slots = available_slots[:3]

        if _redis_client and user_id:
            for s in suggested_slots:
                try:
                    dt_obj = datetime.fromisoformat(s["datetime"])
                    _redis_client.setex(_slot_hold_key(dt_obj), HOLD_TTL_SECONDS, str(user_id))
                except Exception:
                    pass  # If parsing fails or redis unavailable, don't crash

        return suggested_slots
        
    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

def create_booking(service, customer_name, customer_phone, date, time, duration_hours=None, tattoo_price=None, tattoo_description="", user_id=None, thread_id=None):
    """
    Create a new booking in the calendar
    
    Args:
        service: Google Calendar service object
        customer_name: Customer's name
        customer_phone: Customer's phone number
        date: Date in format 'YYYY-MM-DD'
        time: Time in format 'HH:MM'
        duration_hours: Duration of appointment in hours (if not provided, calculated from tattoo_price)
        tattoo_price: Estimated price of tattoo in euros (used to calculate duration if duration_hours not provided)
        tattoo_description: Description of the tattoo
        user_id: Instagram user ID for linking back to conversation
    Returns:
        Created event object or None if error
    """
    # Calculate duration based on price if not explicitly provided
    if duration_hours is None:
        if tattoo_price is None:
            duration_hours = 1  # Default to 1 hour if neither is provided
        else:
            # Formula: price / 50 = number of half-hours, so price / 100 = hours
            raw_duration = tattoo_price / 100
            duration_hours = round_duration_to_5_minutes(raw_duration)
    
    try:
        # Parse date and time
        start_datetime = datetime.strptime(f"{date} {time}", '%Y-%m-%d %H:%M')
        start_datetime = ATHENS_TZ.localize(start_datetime)
        end_datetime = start_datetime + timedelta(hours=duration_hours)
        
        # Build description
        description = f'Πελάτης: {customer_name}\nΤηλέφωνο: {customer_phone}'
        if tattoo_description:
            description += f'\nΤατουάζ: {tattoo_description}'
        if tattoo_price is not None:
            description += f'\nΕκτιμώμενη τιμή: {tattoo_price}€'
            description += f'\nΔιάρκεια: {format_duration_display(duration_hours)}'
        
        
        event = {
            'summary': f'Τατουάζ - {customer_name}',
            'description': description,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': 'Europe/Athens',
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': 'Europe/Athens',
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 60},
                ],
            },
        }
        
        created_event = service.events().insert(calendarId='primary', body=event).execute()

        # Release any temporary hold for this slot, freeing it for others immediately
        if _redis_client:
            try:
                _redis_client.delete(_slot_hold_key(start_datetime))
            except Exception:
                pass  # Non-critical

        return created_event
        
    except Exception as error:
        print(f'An error occurred: {error}')
        return None

def find_booking_by_phone(service, phone_number):
    """
    Find a booking by customer phone number
    
    Args:
        service: Google Calendar service object
        phone_number: Customer's phone number
    
    Returns:
        List of matching events
    """
    try:
        # Search in the next 3 months
        time_min = datetime.now(ATHENS_TZ)
        time_max = time_min + timedelta(days=90)
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            q=phone_number,  # Search query
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Filter events that contain the phone number in description
        matching_events = []
        for event in events:
            if 'description' in event and phone_number in event['description']:
                matching_events.append(event)
        print(f"Found {len(matching_events)} events for phone number {phone_number}", file=log_file)
        return matching_events
        
    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

def cancel_booking(service, event_id):
    """
    Cancel a booking
    
    Args:
        service: Google Calendar service object
        event_id: Google Calendar event ID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return True
    except HttpError as error:
        print(f'An error occurred: {error}')
        return False

def reschedule_booking(service, event_id, new_date, new_time, duration_hours=None, tattoo_price=None):
    """
    Reschedule an existing booking
    
    Args:
        service: Google Calendar service object
        event_id: Google Calendar event ID
        new_date: New date in format 'YYYY-MM-DD'
        new_time: New time in format 'HH:MM'
        duration_hours: Duration of appointment in hours (if not provided, calculated from tattoo_price or existing duration)
        tattoo_price: Estimated price of tattoo in euros (used to calculate duration if duration_hours not provided)
    
    Returns:
        Updated event object or None if error
    """
    try:
        # Get the existing event
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        
        # Calculate duration
        if duration_hours is None:
            if tattoo_price is not None:
                # Formula: price / 50 = number of half-hours, so price / 100 = hours
                raw_duration = tattoo_price / 100
                duration_hours = round_duration_to_5_minutes(raw_duration)
            else:
                # Try to extract duration from existing event
                existing_start = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
                existing_end = datetime.fromisoformat(event['end']['dateTime'].replace('Z', '+00:00'))
                duration_hours = (existing_end - existing_start).total_seconds() / 3600
        
        # Update the time
        start_datetime = datetime.strptime(f"{new_date} {new_time}", '%Y-%m-%d %H:%M')
        start_datetime = ATHENS_TZ.localize(start_datetime)
        end_datetime = start_datetime + timedelta(hours=duration_hours)
        
        event['start'] = {
            'dateTime': start_datetime.isoformat(),
            'timeZone': 'Europe/Athens',
        }
        event['end'] = {
            'dateTime': end_datetime.isoformat(),
            'timeZone': 'Europe/Athens',
        }
        
        # Update description if price is provided
        if tattoo_price is not None and 'description' in event:
            # Update or add price and duration info
            description_lines = event['description'].split('\n')
            new_lines = []
            price_added = False
            duration_added = False
            
            for line in description_lines:
                if line.startswith('Εκτιμώμενη τιμή:'):
                    new_lines.append(f'Εκτιμώμενη τιμή: {tattoo_price}€')
                    price_added = True
                elif line.startswith('Διάρκεια:'):
                    new_lines.append(f'Διάρκεια: {format_duration_display(duration_hours)}')
                    duration_added = True
                else:
                    new_lines.append(line)
            
            if not price_added:
                new_lines.append(f'Εκτιμώμενη τιμή: {tattoo_price}€')
            if not duration_added:
                new_lines.append(f'Διάρκεια: {format_duration_display(duration_hours)}')
            
            event['description'] = '\n'.join(new_lines)
        
        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=event
        ).execute()
        
        return updated_event
        
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def format_available_slots_message(available_slots):
    """
    Format available slots into a user-friendly message in Greek
    
    Args:
        available_slots: List of available slot dictionaries
    
    Returns:
        Formatted string message
    """
    if not available_slots:
        return "Δυστυχώς δεν υπάρχουν διαθέσιμες ώρες για τις ημερομηνίες που ζητήσατε."
    
    message = "Διαθέσιμες ώρες:\n\n"
    
    # Group by date
    dates = {}
    for slot in available_slots:
        date = slot['date']
        if date not in dates:
            dates[date] = []
        dates[date].append(slot['start_time'])
    
    # Format each date
    for date, times in dates.items():
        # Convert date to Greek format
        dt = datetime.strptime(date, '%Y-%m-%d')
        days_greek = ['Δευτέρα', 'Τρίτη', 'Τετάρτη', 'Πέμπτη', 'Παρασκευή', 'Σάββατο', 'Κυριακή']
        months_greek = ['Ιανουαρίου', 'Φεβρουαρίου', 'Μαρτίου', 'Απριλίου', 'Μαΐου', 'Ιουνίου',
                       'Ιουλίου', 'Αυγούστου', 'Σεπτεμβρίου', 'Οκτωβρίου', 'Νοεμβρίου', 'Δεκεμβρίου']
        
        day_name = days_greek[dt.weekday()]
        month_name = months_greek[dt.month - 1]
        
        message += f"📅 {day_name}, {dt.day} {month_name}:\n"
        message += f"   ⏰ {', '.join(times[:3])}"  # Show first 3 times
        if len(times) > 3:
            message += f" και άλλες {len(times) - 3}"
        message += "\n\n"
    
    return message.strip()

def format_duration_display(duration_hours):
    """
    Format duration for display in hours and minutes
    
    Args:
        duration_hours: Duration in hours (can be decimal)
    
    Returns:
        Formatted string like "1 ώρα και 30 λεπτά" or "2 ώρες"
    """
    total_minutes = int(duration_hours * 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    if hours == 0:
        return f"{minutes} λεπτά"
    elif minutes == 0:
        if hours == 1:
            return "1 ώρα"
        else:
            return f"{hours} ώρες"
    else:
        if hours == 1:
            return f"1 ώρα και {minutes} λεπτά"
        else:
            return f"{hours} ώρες και {minutes} λεπτά"