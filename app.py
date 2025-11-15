import json
from flask import Flask, request
import time
import os
import redis
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from datetime import datetime
import threading
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from calendar_functions import (
    check_availability,
    create_booking,
    find_booking_by_phone,
    cancel_booking,
    reschedule_booking,
    format_available_slots_message,
    authenticate_google,
    get_calendar_service
)
import random

# Import new services and utilities
from src.services.openai_service import OpenAIService
from src.services.messaging_service import MessagingService
from src.utils.logging_utils import configure_logging, get_logger, LoggerAdapter
from src.utils.redis_utils import init_redis_client, test_redis_connection
from src.utils.constants import (
    MAX_HISTORY_LENGTH,
    GRACE_WINDOW_SECONDS,
    MAX_RETRY_ATTEMPTS,
    MESSAGE_MAX_LENGTH,
    PROCESSING_LOCK_TTL,
    CONVERSATION_TTL_SECONDS,
    QUEUE_TTL_SECONDS,
    MUTE_DURATION_SECONDS,
    OPENAI_MODEL_DEFAULT,
    OPENAI_MODEL_VISION,
    OPENAI_MODEL_CLASSIFY,
    OPENAI_TEMPERATURE_DEFAULT,
    OPENAI_TEMPERATURE_PRICING,
    OPENAI_TEMPERATURE_CLASSIFY,
    CONVERSATIONS_INDEX_NAME,
    PRICING_INDEX_NAME,
    SIMILARITY_THRESHOLD,
    TOP_K_SIMILAR,
    INTENT_PRIORITIES,
    DEFAULT_ADMIN_SENDER_IDS,
    REACTION_BOT_SENDER_ID_DEFAULT,
    PRICING_PROMPT_FILE,
    BOOKING_PROMPT_FILE,
    INFORMATION_PROMPT_FILE,
    FOLLOWUP_PROMPT_FILE,
    CLASSIFICATION_PROMPT_FILE
)

# Load environment variables
load_dotenv()

# Configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Configure structured logging
configure_logging(debug=DEBUG)
logger = get_logger("app")

# Create backwards-compatible log file adapter
log_file = LoggerAdapter("app")

# Initialize Flask app
app = Flask(__name__)

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("REDIS_URL") or f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
)

# Initialize Redis client using utility
redis_client = init_redis_client()

# Test Redis connection on startup (only in DEBUG)
if DEBUG:
    test_redis_connection(redis_client)

# Environment-specific configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Admin configuration (prefer env vars, fallback to defaults)
admin_ids_str = os.getenv("ADMIN_SENDER_IDS", ",".join(DEFAULT_ADMIN_SENDER_IDS))
ADMIN_SENDER_IDS = set(filter(None, [s.strip() for s in admin_ids_str.split(",")]))
REACTION_BOT_SENDER_ID = os.getenv("REACTION_BOT_SENDER_ID", REACTION_BOT_SENDER_ID_DEFAULT)

# Initialize external services
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Initialize OpenAI service (replaces direct client)
openai_service = OpenAIService()

# Initialize Instagram messaging service
messaging_service = MessagingService()

# Initialize Google Calendar
creds = authenticate_google()
service = get_calendar_service(creds)

logger.info("application_initialized", debug=DEBUG)

@app.route('/')
def hello_world():
    return "Hello world!!!!!!!"

@app.route('/health')
def health_check():
    """Health check endpoint that includes Redis status"""
    status = {
        "status": "healthy",
        "redis": "disconnected",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        if test_redis_connection():
            status["redis"] = "connected"
            return status, 200
        else:
            status["status"] = "degraded"
            return status, 503
    except Exception as e:
        status["status"] = "unhealthy"
        status["error"] = str(e)
        return status, 503


@app.route('/privacy_policy')
def privacy_policy():
    try:
        with open('./privacy_policy.html', 'rb') as f:
            privacy_policy_text = f.read()
        return privacy_policy_text
    except Exception:
        return "Privacy Policy not available.", 404

@app.route('/terms_of_service')
def terms():
    try:
        with open('./terms.html', 'rb') as f:
            terms_txt = f.read()
        return terms_txt
    except Exception:
        return "Terms of Service not available.", 404

@app.route('/webhook', methods = ['GET', 'POST'])
@limiter.limit("100 per hour")  # Stricter limit for webhook endpoint
def webhook():
    if request.method == 'POST':
        try:
            if DEBUG:
                print(f"POST request to /webhook from {request.remote_addr}", file=log_file)
                print("Request headers:", file=log_file)
                print(json.dumps(dict(request.headers), indent=4), file=log_file)
            
            request_data = request.json
            if DEBUG:
                print("Request data:", file=log_file)
                print(json.dumps(request_data, indent=4, ensure_ascii=False), file=log_file)
                log_file.flush()
            
            # Extract the PSID from the webhook payload
            if "entry" in request_data and request_data["entry"]:
                entry = request_data["entry"][0]
                if "messaging" in entry and entry["messaging"]:
                    messaging = entry["messaging"][0]
                    if "sender" in messaging and "id" in messaging["sender"]:
                        sender_id = messaging["sender"]["id"]
                        recipient_id = messaging["recipient"]["id"]
                        
                        if sender_id in ADMIN_SENDER_IDS:
                            # Handle reactions (human-in-the-loop override)
                            if "reaction" in messaging and sender_id == REACTION_BOT_SENDER_ID:
                                reaction_data = messaging["reaction"]
                                if reaction_data.get("emoji") == "❤":
                                    mute_user(recipient_id)
                                    clear_message_queue(recipient_id)
                                    redis_client.delete(f"scheduled:{recipient_id}")
                                    return "EVENT_RECEIVED", 200
                            # Check if this is an image message
                            has_image = False
                            if "message" in messaging and "attachments" in messaging["message"]:
                                attachments = messaging["message"]["attachments"]
                                
                                # Set pending counter instead of a simple flag
                                pending_key = f"images_pending:{sender_id}"
                                # Get current pending count (if exists)
                                pending_count = int(redis_client.get(pending_key) or "0")
                                # Add number of new images to process
                                new_pending = pending_count + len([a for a in attachments if a["type"] == "image"])
                                # Update the counter with 1 hour expiration
                                redis_client.setex(pending_key, 60*60, str(new_pending))

                                for i, attachment in enumerate(attachments, start=1):
                                    if attachment["type"] == "image":
                                        has_image = True
                                        image_url = attachment["payload"]["url"]
                                        try:
                                            image_path = download_image(image_url, sender_id)
                                            image_analysis = f"Εικόνα {i}: " + get_image_analysis_reply(image_path) + "\n"                                      
                                            redis_client.rpush(f"image_analysis:{sender_id}", image_analysis)
                                            redis_client.expire(f"image_analysis:{sender_id}", QUEUE_TTL_SECONDS)
                                            os.remove(image_path)
                                            redis_client.decr(pending_key)
                                        except Exception as img_error:
                                            print(f"Error processing image {i}: {str(img_error)}", file=log_file)
                                            log_file.flush()
                                            continue
                            queue_user_message(sender_id, messaging, has_image)            
                            return "EVENT_RECEIVED", 200
                    else:
                        print("Missing sender.id in messaging", file=log_file)
                else:
                    print("Missing messaging in entry", file=log_file)
            else:
                print("Missing entry in request_data", file=log_file)
            
            log_file.flush()
            return "INVALID_PAYLOAD", 400
            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error in webhook: {str(e)}", file=log_file)
            print("Raw request data:", request.get_data(), file=log_file)
            log_file.flush()
            return "INVALID_JSON", 400
            
        except redis.exceptions.ConnectionError as e:
            print(f"Redis Connection Error in webhook: {str(e)}", file=log_file)
            log_file.flush()
            return "REDIS_ERROR", 503
            
        except Exception as e:
            print(f"Unexpected Error in webhook: {str(e)}", file=log_file)
            import traceback
            print(traceback.format_exc(), file=log_file)
            log_file.flush()
            return "ERROR", 500
            
    elif request.method == 'GET':
        hub_mode = request.args.get('hub.mode')
        hub_challenge = request.args.get('hub.challenge')
        hub_verify_token = request.args.get('hub.verify_token')
        if hub_challenge:
            return hub_challenge
        else:
            return "<p>This is GET request, Hello Webhook!</p>"

def retrieve_thread_id(message_id):
    url = f"https://graph.facebook.com/v19.0/{message_id}?fields=thread&access_token={USER_ACCESS_TOKEN}"
    response = requests.get(url)
    data = response.json()
    return data["thread"]["id"]

def send_instagram_message(recipient_id, message_text):
    """
    Send a message to an Instagram user using the Graph API
    This will only work if the user has messaged your business account first

    NOTE: This is now a wrapper around MessagingService for backwards compatibility
    """
    return messaging_service.send_message(recipient_id, message_text)

# Define OpenAI function schemas
CALENDAR_FUNCTIONS = [
    {
        "name": "check_calendar_availability",
        "description": "Check available time slots in the calendar for tattoo appointments",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date to check availability (format: YYYY-MM-DD)"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date to check availability (format: YYYY-MM-DD). Optional, defaults to start_date"
                },
                "duration_hours": {
                    "type": "number",
                    "description": "Duration of the appointment in hours (if not provided, calculated from tattoo_price)"
                },
                "tattoo_price": {
                    "type": "number",
                    "description": "Estimated price of tattoo in euros (used to calculate duration: price/100 = hours)"
                },
                "user_id": {
                    "type": "string",
                    "description": "Instagram user ID (required for temporary slot holds)"
                },
                "preferred_time": {
                    "type": "string",
                    "description": "Preferred appointment time (format: HH:MM). Suggestions will start no earlier than this time on the first requested day"
                }
            },
            "required": ["start_date"]
        }
    },
    {
        "name": "create_tattoo_booking",
        "description": "Create a new tattoo appointment booking",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Customer's full name"
                },
                "customer_phone": {
                    "type": "string", 
                    "description": "Customer's phone number"
                },
                "date": {
                    "type": "string",
                    "description": "Appointment date (format: YYYY-MM-DD)"
                },
                "time": {
                    "type": "string",
                    "description": "Appointment time (format: HH:MM)"
                },
                "duration_hours": {
                    "type": "number",
                    "description": "Duration of the appointment in hours (if not provided, calculated from tattoo_price)"
                },
                "tattoo_price": {
                    "type": "number",
                    "description": "Estimated price of tattoo in euros (used to calculate duration: price/100 = hours)"
                },
                "tattoo_description": {
                    "type": "string",
                    "description": "Description of the tattoo design/style"
                },
                "user_id": {
                    "type": "string",
                    "description": "Instagram user ID (thread owner)"
                }
            },
            "required": ["customer_name", "customer_phone", "date", "time", "user_id"]
        }
    },
    {
        "name": "find_customer_booking",
        "description": "Find existing bookings by customer phone number",
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "Customer's phone number to search for"
                }
            },
            "required": ["phone_number"]
        }
    },
    {
        "name": "cancel_tattoo_booking",
        "description": "Cancel an existing tattoo appointment",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "Google Calendar event ID of the booking to cancel"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "reschedule_tattoo_booking",
        "description": "Reschedule an existing tattoo appointment",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "Google Calendar event ID of the booking to reschedule"
                },
                "new_date": {
                    "type": "string",
                    "description": "New appointment date (format: YYYY-MM-DD)"
                },
                "new_time": {
                    "type": "string",
                    "description": "New appointment time (format: HH:MM)"
                },
                "duration_hours": {
                    "type": "number",
                    "description": "Duration of the appointment in hours (if not provided, calculated from tattoo_price)"
                },
                "tattoo_price": {
                    "type": "number",
                    "description": "Estimated price of tattoo in euros (used to calculate duration: price/100 = hours)"
                }
            },
            "required": ["event_id", "new_date", "new_time"]
        }
    }
]

def extract_phone_number_from_context(context):
    """Extract phone number from conversation context"""
    import re
    
    # Greek phone number patterns
    phone_patterns = [
        r'\b69\d{8}\b',  # Mobile numbers starting with 69
        r'\b\+30\s?69\d{8}\b',  # With country code
        r'\b\+30\s?\d{10}\b',  # Any Greek number with country code
        r'\b21\d{8}\b',  # Athens landlines
        r'\b\d{10}\b'  # Any 10-digit number
    ]
    
    # Search through all conversation messages
    for entry in reversed(context):  # Start from most recent
        if entry.get("content"):
            content = entry["content"]
            for pattern in phone_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Clean and return the first valid phone number
                    phone = matches[0].replace('+30', '').replace(' ', '')
                    if len(phone) == 10 and phone.isdigit():
                        return phone
    
    return None

def execute_calendar_function(function_name, arguments):
    """Execute calendar functions and return results"""
    try:
        if function_name == "check_calendar_availability":
            slots = check_availability(
                service,
                arguments.get("start_date"),
                arguments.get("end_date"),
                arguments.get("duration_hours"),
                arguments.get("tattoo_price"),
                arguments.get("user_id"),
                arguments.get("preferred_time")
            )
            return {"status": "success", "slots": slots, "message": format_available_slots_message(slots)}
        
        elif function_name == "create_tattoo_booking":
            # Check if user has a hold and use it instead of creating new booking
            user_id = arguments.get('user_id', '')
            # Create new booking normally
            event = create_booking(
                service,
                arguments["customer_name"],
                arguments["customer_phone"],
                arguments["date"],
                arguments["time"],
                arguments.get("duration_hours"),
                arguments.get("tattoo_price"),
                arguments.get("tattoo_description", ""),
                arguments.get("thread_id", user_id)  # Use user_id as fallback for thread_id
            )
            if event:
                return {"status": "success", "event_id": event['id'], "message": "Το ραντεβού δημιουργήθηκε επιτυχώς!"}
            else:
                return {"status": "error", "message": "Δυστυχώς δεν μπόρεσε να δημιουργηθεί το ραντεβού."}
        
        elif function_name == "find_customer_booking":
            events = find_booking_by_phone(service, arguments["phone_number"])
            if events:
                return {"status": "success", "events": events, "count": len(events)}
            else:
                return {"status": "not_found", "message": "Δεν βρέθηκαν ραντεβού με αυτό το τηλέφωνο."}
        
        elif function_name == "cancel_tattoo_booking":
            success = cancel_booking(service, arguments["event_id"])
            if success:
                return {"status": "success", "message": "Το ραντεβού ακυρώθηκε επιτυχώς."}
            else:
                return {"status": "error", "message": "Δυστυχώς δεν μπόρεσε να ακυρωθεί το ραντεβού."}
        
        elif function_name == "reschedule_tattoo_booking":
            event = reschedule_booking(
                service,
                arguments["event_id"],
                arguments["new_date"],
                arguments["new_time"],
                arguments.get("duration_hours"),
                arguments.get("tattoo_price")
            )
            if event:
                return {"status": "success", "message": "Το ραντεβού μεταφέρθηκε επιτυχώς!"}
            else:
                return {"status": "error", "message": "Δυστυχώς δεν μπόρεσε να μεταφερθεί το ραντεβού."}
    
    except Exception as e:
        return {"status": "error", "message": f"Σφάλμα: {str(e)}"}

def get_convo_context(user_id):
    """Get conversation context with enhanced Redis Cloud error handling"""
    global redis_client
    try:
        raw = redis_client.get(f"chat:{user_id}")
        return json.loads(raw) if raw else []
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error in get_convo_context: {str(e)}", file=log_file)
        log_file.flush()
        # Try to reconnect once
        try:
            redis_client = init_redis_client()
            raw = redis_client.get(f"chat:{user_id}")
            return json.loads(raw) if raw else []
        except Exception as reconnect_error:
            print(f"Redis reconnection failed: {str(reconnect_error)}", file=log_file)
            log_file.flush()
            return []
    except redis.exceptions.TimeoutError as e:
        print(f"Redis timeout error in get_convo_context: {str(e)}", file=log_file)
        log_file.flush()
        return []
    except Exception as e:
        print(f"Unexpected error in get_convo_context: {str(e)}", file=log_file)
        log_file.flush()
        return []

def save_convo_context(user_id, new_entry):
    """Save conversation context with enhanced Redis Cloud error handling"""
    global redis_client
    try:
        context = get_convo_context(user_id)
        context.append(new_entry)
        context = context[-MAX_HISTORY_LENGTH:]  # keep last messages according to MAX_HISTORY_LENGTH
        redis_client.setex(f"chat:{user_id}", CONVERSATION_TTL_SECONDS, json.dumps(context))
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error in save_convo_context: {str(e)}", file=log_file)
        log_file.flush()
        # Try to reconnect and save again
        try:
            redis_client = init_redis_client()
            context = get_convo_context(user_id)
            context.append(new_entry)
            context = context[-MAX_HISTORY_LENGTH:]
            redis_client.setex(f"chat:{user_id}", CONVERSATION_TTL_SECONDS, json.dumps(context))
        except Exception as reconnect_error:
            print(f"Failed to save conversation after reconnection: {str(reconnect_error)}", file=log_file)
            log_file.flush()
    except Exception as e:
        print(f"Error saving conversation context: {str(e)}", file=log_file)
        log_file.flush()

def download_image(image_url, user_id):
    """
    Download an image from Instagram

    NOTE: This is now a wrapper around MessagingService for backwards compatibility
    """
    return messaging_service.download_image(image_url, user_id)

def get_openai_call_for_intent(context, intents_list, user_id, text_messages):
    """
    Returns a custom OpenAI API call (or parameters) based on the intents.
    Handles multiple intents by prioritizing and responding to the most important one.
    """
    openai_model = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-4o")
    temperature = 1.0
    extra_kwargs = {}
    tools = None  # Initialize tools

    conversations_index = pc.Index(CONVERSATIONS_INDEX_NAME)
    pricing_index = pc.Index(PRICING_INDEX_NAME)

    # Define intent priority order (higher priority = lower number)
    intent_priority = {
        "pricing": 1,
        "studio_information": 3,
        "booking_request": 2,
        "follow_up": 4,
        "other": 5
    }

    # Sort intents by priority and confidence
    sorted_intents = sorted(
        intents_list,
        key=lambda x: (
            intent_priority.get(x.get("primary", "other"), 999),
            -x.get("confidence", 0)  # Negative confidence for descending sort
        )
    )
    
    # Get the highest priority intent
    primary_intent = sorted_intents[0] if sorted_intents else {"primary": "other"}
    
    # Identify other intents that were detected
    other_intents = [intent for intent in sorted_intents[1:] if intent.get("primary") != primary_intent.get("primary")]
    
    # Retrieve similar conversations for the primary intent
    retrieved_examples = retrieve_similar_conversations(text_messages, conversations_index)

    # Default prompt
    prompt = """
                Απαντάς σε DM πελατών του 210tattoo. Δεν είσαι chatbot — είσαι μέλος της ομάδας. Η δουλειά σου είναι να απαντάς 100% όπως έχεις μάθει από τα παραδείγματα του training.

                Χρησιμοποιείς ακριβώς τις ίδιες φράσεις, emoji, τιμές και ύφος. Δεν αυτοσχεδιάζεις, δεν προσθέτεις δικά σου λόγια, δεν εξηγείς τεχνικά. Αν δεν είσαι σίγουρος/η, βασίζεσαι αποκλειστικά σε όσα έχεις μάθει.

                Γράφεις πάντα στα ελληνικά.
                Πάντα στο τέλος του μηνύματος να βάζεις τα emoji "❤️🐼".

                *Προτεραιότητα:*  
                Δεν απαντάς με δικά σου λόγια. Δεν λες τίποτα που δεν έχει φανεί στα παραδείγματα.

            """
    
    # Add multi-intent handling instructions if multiple intents are detected
    if other_intents:
        intent_acknowledgment = "\n\n**Σημαντικό:** Ο πελάτης έκανε πολλαπλές ερωτήσεις. "
        
        # Special handling for pricing + booking combination
        if primary_intent["primary"] == "pricing" and any(intent.get("primary") == "booking_request" for intent in other_intents):
            intent_acknowledgment += "Απάντησε ΜΟΝΟ στην ερώτηση για την τιμή. Πες ότι αφού συμφωνήσουμε στο τατουάζ και την τιμή, μετά θα συζητήσουμε για ραντεβού."
        # Special handling for booking when pricing hasn't been discussed
        elif primary_intent["primary"] == "booking_request" and any(intent.get("primary") == "pricing" for intent in other_intents):
            intent_acknowledgment += "Απάντησε ΠΡΩΤΑ στην ερώτηση για την τιμή, και πες ότι μετά θα συζητήσουμε για ραντεβού."
            # Switch to pricing as primary intent
            primary_intent = next((intent for intent in other_intents if intent.get("primary") == "pricing"), primary_intent)
        else:
            intent_acknowledgment += f"Εστίασε στην κύρια ερώτηση ({primary_intent['primary']}) και πες ότι θα απαντήσεις στα υπόλοιπα στη συνέχεια."
        
        prompt += intent_acknowledgment

    pricing_examples_text = ""

    if primary_intent["primary"] == "pricing":
        with open('./prompts/pricing.txt', 'r', encoding='utf-8') as f:
            prompt = f.read()

        # Handle new_quote_with_image
        if primary_intent.get("subcategory") == "new_quote_image":
            # Extract image analyses from complete_message
            image_analysis = redis_client.lrange(f"image_analysis:{user_id}", 0, -1)
            # pricing_examples = []

            # for idx, analysis in enumerate(image_analysis, start=1):
            #     # Query pricing index for the closest match
            #     query_embedding = model.encode(analysis).tolist()
            #     results = pricing_index.query(
            #         vector=query_embedding,
            #         top_k=1,
            #         include_metadata=True
            #     )
            #     if results["matches"]:
            #         match = results["matches"][0]
            #         meta = match["metadata"]
            #         pricing_examples.append(
            #             f"Για το τατουάζ {analysis}, προτεινόμενο παράδειγμα τιμολόγησης: "
            #             f"(παρόμοια περιγραφή: {meta.get('description', '')}, διαστάσεις: {meta.get('width_cm', '?')}x{meta.get('height_cm', '?')}cm, εμβαδόν: {meta.get('area_cm2', '?')}cm², τιμή που δόθηκε: {meta.get('price', '?')}€)"
            #         )
            
            # if pricing_examples:
            #     pricing_examples_text = "\n\n# Παρόμοια παραδείγματα τιμολόγησης:\n" + "\n".join(pricing_examples)
            # else:
            #     pricing_examples_text = ""
            
            if image_analysis:
                analyses_text = "\n\n# Ανάλυση εικόνων (raw):\n" + "\n".join(image_analysis)
            else:
                analyses_text = ""
            
            prompt += analyses_text

            prompt += """
                            - Δεν επινοείς τιμές. Δεν λες ποτέ \"περίπου\", \"ξεκινάει από\", \"ανάλογα\".
                            - Χρησιμοποίησε τις τιμές h, w, ink και D που εμφανίζονται μέσα στο κείμενο της/των ανάλυσης/-εων εικόνας (μορφή h=.. | w=.. | ink=.. | D=..).
                            - Υπολόγισε για ΚΑΘΕ τατουάζ την τιμή P_i με τον τύπο:
                              P_i = max(45, floor((h × w × D × (1 + 0.3 × ink)) / 5) × 5)

                            - ***Αν έχεις 1 μόνο τατουάζ:***
                              • Δώσε ΠΑΝΤΑ **δύο** τιμές Χ και Ψ.
                                ◦ Αν P_1 < 90€ → Ψ = Χ + 5€
                                ◦ Αν P_1 ≥ 90€ → Ψ = Χ + έως 10€ (διάλεξε τιμή ώστε και οι δύο να είναι πολλαπλάσια του 5).
                              • Μην εξηγήσεις τον υπολογισμό.
                              • Απάντησε αποκλειστικά με το παρακάτω template (χωρίς πρόσθετα emoji ή προτάσεις):
                                Καλησπέρα ❤️🐼 , θα σας εκτυπώσουμε από κοντά 2 μεγέθη ένα στα *Χ€* και ένα στα *Ψ€* για να διαλέξουμε μαζί ποιο σας ταιριάζει περισσότερο 😊 Οι ώρες μας γεμίζουν πολύ γρήγορα αυτές τις μέρες! 😊 Θέλετε να σας κλείσουμε ραντεβού? 

                            - ***Αν έχεις 2 ή περισσότερα τατουάζ:***
                              • Υπολόγισε P_i για κάθε τατουάζ.
                              • Υπολόγισε το άθροισμα S = Σ P_i.
                              • Εφάρμοσε έκπτωση 10%: T = floor((S × 0.9) / 5) × 5 (στρογγυλοποίησε προς τα κάτω στο πλησιέστερο πολλαπλάσιο του 5).
                              • Μην αναφέρεις επιμέρους τιμές ή τον τρόπο που έγινε η έκπτωση – μόνο το τελικό ποσό Τ.
                              • Απάντησε αποκλειστικά με το παρακάτω template:
                                Καλησπέρα ❤️🐼 , το συνολικό κόστος για τα τατουάζ είναι *Τ€* 😊 Οι ώρες μας γεμίζουν πολύ γρήγορα αυτές τις μέρες! 😊 Θέλετε να σας κλείσουμε ραντεβού? 

                            - Μην εξηγήσεις τον υπολογισμό ή τα ενδιάμεσα βήματα.
                        """
            temperature = 0.3
        elif primary_intent.get("subcategory") == "new_quote_no_image":
            prompt += "Αν δεν έχει στείλει κάτι ξεκάθαρο σε περιγραφή, ρωτάς ευγενικά να σου στείλει κάποια φωτογραφία ή περιγραφή του τατουάζ που θέλει."
        
        # If booking was also mentioned, add specific instruction
        if any(intent.get("primary") == "booking_request" for intent in other_intents):
            prompt += "\n\nΕπίσης, αφού δώσεις την τιμή, πες ότι μόλις συμφωνήσουμε στο τατουάζ και την τιμή, θα κανονίσουμε το ραντεβού σου."

        # Add retrieved conversation examples
        examples_text = ""
        for i, example in enumerate(retrieved_examples):
            examples_text += f"\nΠαράδειγμα {i+1}:\nΕρώτηση: {example['query']}\nΑπάντηση: {example['response']}\n"

        print(pricing_examples_text, file=log_file)
        # Combine everything
        prompt += pricing_examples_text
        prompt += f"\n\n## Παρόμοιες συνομιλίες από το παρελθόν:{examples_text}\n\n"
        prompt += "\nΧρησιμοποίησε τα παραδείγματα με σκοπο να προσεγγισεις τον τροπο που απαντησαν οι ανθρωποι στην ομαδα μας. Αν δεν μπορεις να βρεις κατι παρομοιο, απαντα με τον τροπο που εχεις μαθει απο τα παραδειγματα του training."

    elif primary_intent["primary"] == "booking_request":
        with open('./prompts/booking.txt', 'r', encoding='utf-8') as f:
            prompt = f.read()
            
        # Enable function calling for booking requests
        tools = [{"type": "function", "function": func} for func in CALENDAR_FUNCTIONS]
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        prompt += f"\n\n**Σημερινή ημερομηνία: {current_date}**"
        
        # Add specific function calling instructions
        prompt += "\n\n**ΣΗΜΑΝΤΙΚΟ για τις λειτουργίες ημερολογίου:**"
        prompt += f"\n**ΠΑΝΤΑ συμπερίλαβε το user_id: '{user_id}' σε όλες τις κλήσεις συναρτήσεων**"
        
        # If 'available_slots' is detected among intents, prioritize it as it's more specific
        # and contains valuable date information that 'new_appointment' might lack.
        if any(i.get("subcategory") == "available_slots" for i in sorted_intents if i.get("primary") == "booking_request"):
            # Find the 'available_slots' intent and make it the primary one to be processed
            primary_intent = next((i for i in sorted_intents if i.get("subcategory") == "available_slots"), primary_intent)
        
        if primary_intent.get("subcategory") == "new_appointment":
            prompt +='''
            - Όταν ο πελάτης ζητάει διαθέσιμες ώρες, χρησιμοποίησε το check_calendar_availability
            - Αν έχετε συζητήσει τιμή για το τατουάζ, χρησιμοποίησε το tattoo_price για αυτόματο υπολογισμό διάρκειας
            - Όταν έχετε συμφωνήσει σε ώρα και έχεις όνομα/τηλέφωνο, χρησιμοποίησε το create_tattoo_booking
            - **ΣΗΜΑΝΤΙΚΟ: ΜΗΝ αναφέρεις πόση είναι η εκτιμώμενη διάρκεια του ραντεβού, εκτός αν ρωτήσει ο πελάτης**
            - **ΣΗΜΑΝΤΙΚΟ: ΜΗΝ αναφέρεις το κόστος που συμφωνήσατε**
            - Αν λείπουν στοιχεία (όνομα, τηλέφωνο, ημερομηνία, ώρα), ρώτα ευγενικά
            '''
        elif primary_intent.get("subcategory") == "provide_details":
            prompt +='''
            - Αν το μήνυμα περιέχει όνομα και τηλέφωνο, χρησιμοποίησε το create_tattoo_booking για το datetime που συμφωνήσατε
            - **ΣΗΜΑΝΤΙΚΟ: ΜΗΝ αναφέρεις πόση είναι η εκτιμώμενη διάρκεια του ραντεβού, εκτός αν ρωτήσει ο πελάτης**
            - **ΣΗΜΑΝΤΙΚΟ: ΜΗΝ αναφέρεις το κόστος που συμφωνήσατε**
            - Αν λείπουν στοιχεία (ημερομηνία, ώρα), ρώτα ευγενικά
            '''
        elif primary_intent.get("subcategory") == "reschedule_appointment":
            prompt +='''
            - Πρώτα χρησιμοποίησε το find_customer_booking για να βρεις το υπάρχον ραντεβού
            - Μετά ρώτα για νέα ημερομηνία/ώρα και χρησιμοποίησε το reschedule_tattoo_booking
            - Αν έχετε συζητήσει νέα τιμή, χρησιμοποίησε το tattoo_price για αυτόματο υπολογισμό διάρκειας
            '''
        elif primary_intent.get("subcategory") == "cancel_appointment":
            # Try to extract phone number from context
            phone_number = extract_phone_number_from_context(context)
            
            if phone_number:
                prompt += f'''
            - ΣΗΜΑΝΤΙΚΟ: Χρησιμοποίησε το τηλέφωνο {phone_number} για να βρεις τα ραντεβού του πελάτη
            - Καλέσε find_customer_booking με phone_number: "{phone_number}"
            - Αν βρεθούν ραντεβού, κάλεσε ΑΜΕΣΩΣ cancel_tattoo_booking με το event_id του πιο πρόσφατου ραντεβού
            - Αν υπάρχουν πολλά ραντεβού, ακύρωσε το πιο πρόσφατο και ενημέρωσε τον πελάτη
            '''
            else:
                prompt += '''
            - ΣΗΜΑΝΤΙΚΟ: Για ακυρώσεις ραντεβού χρειάζεσαι τον αριθμό τηλεφώνου του πελάτη
            - Δεν βρέθηκε τηλέφωνο στη συνομιλία - ρώτησε τον πελάτη για τον αριθμό του
            - Όταν δώσει τηλέφωνο, χρησιμοποίησε find_customer_booking για να βρεις τα ραντεβού του
            - Στη συνέχεια κάλεσε cancel_tattoo_booking με το event_id του ραντεβού που θέλει να ακυρώσει
            '''
        elif primary_intent.get("subcategory") == "available_slots":
            # Extract dates from the intent classification
            start_date = primary_intent.get("start_date")
            end_date = primary_intent.get("end_date")
            
            # Convert DD/MM/YYYY to YYYY-MM-DD format if dates are provided
            if start_date and "/" in start_date:
                day, month, year = start_date.split("/")
                start_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            if end_date and "/" in end_date:
                day, month, year = end_date.split("/")
                end_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            prompt +='''
            - Χρησιμοποίησε το check_calendar_availability για να δεις διαθέσιμες ώρες
            - Αν έχετε συζητήσει τιμή για το τατουάζ, χρησιμοποίησε το tattoo_price για αυτόματο υπολογισμό διάρκειας
            - Χρησιμοποίησε τη φράση "Για άμεσα έχουμε availiable_slot" για να πεις την πρώτη διαθέσιμη ώρα αν ρωτάει γενικά για διαθέσιμες ώρες
            - Στην απάντηση σου, να περιλαμβάνεις την πλήρη ημερομηνία της ημέρας π.χ. για την "Τετάρτη 5/6 έχουμε ..."
            '''
            
            if start_date and end_date:
                prompt += f'''
            - ΣΗΜΑΝΤΙΚΟ: Έχουν εξαχθεί οι ημερομηνίες από το μήνυμα
            - Χρησιμοποίησε start_date: {start_date}
            - Χρησιμοποίησε end_date: {end_date}
            '''
            else:
                prompt += '''
            - Αν δεν έδωσε ημερομηνία, πρότεινε το πρώτο διαθέσιμο ραντεβού που θα βρεις που θα είναι τουλάχιστον 3 ώρες από τώρα
            - ΣΗΜΑΝΤΙΚΟ: Χρησιμοποίησε την σημερινή ημερομηνία ως start_date
            - Για end_date, χρησιμοποίησε 7 μέρες από σήμερα
            '''
            
            prompt += '''
            - Χρησιμοποίησε πάντα το format YYYY-MM-DD για τις ημερομηνίες
            - ΣΗΜΑΝΤΙΚΟ: ΜΗΝ αναφέρεις την end_date, το κόστος ή την εκτιμώμενη διάρκεια του ραντεβού στην απάντησή σου προς τον πελάτη
            - Απλά πες τις διαθέσιμες ώρες χωρίς να αναφέρεις το εύρος ημερομηνιών που έψαξες
            '''
        
        # Add retrieved conversation examples
        examples_text = ""
        for i, example in enumerate(retrieved_examples):
            examples_text += f"\nΠαράδειγμα {i+1}:\nΕρώτηση: {example['query']}\nΑπάντηση: {example['response']}\n"

        # Combine everything
        prompt += f"\n\n## Παρόμοιες συνομιλίες από το παρελθόν:{examples_text}\n\n"
        prompt += "\nΧρησιμοποίησε τα παραδείγματα με σκοπο να προσεγγισεις τον τροπο που απαντησαν οι ανθρωποι στην ομαδα μας."
    
    elif primary_intent["primary"] == "studio_information":
        with open('./prompts/information.txt', 'r', encoding='utf-8') as f:
            prompt = f.read()
                # Add retrieved conversation examples
        examples_text = ""
        for i, example in enumerate(retrieved_examples):
            examples_text += f"\nΠαράδειγμα {i+1}:\nΕρώτηση: {example['query']}\nΑπάντηση: {example['response']}\n"

        # Combine everything
        prompt += f"\n\n## Παρόμοιες συνομιλίες από το παρελθόν:{examples_text}\n\n"
        prompt += "\nΧρησιμοποίησε τα παραδείγματα με σκοπο να προσεγγισεις τον τροπο που απαντησαν οι ανθρωποι στην ομαδα μας. Αν δεν μπορεις να βρεις κατι παρομοιο, απαντα με τον τροπο που εχεις μαθει απο τα παραδειγματα του training."

    elif primary_intent["primary"] == "follow_up":
        with open('./prompts/follow_up.txt', 'r', encoding='utf-8') as f:
            prompt = f.read()
            # Add retrieved conversation examples
        examples_text = ""
        for i, example in enumerate(retrieved_examples):
            examples_text += f"\nΠαράδειγμα {i+1}:\nΕρώτηση: {example['query']}\nΑπάντηση: {example['response']}\n"
        prompt += f"\n\n## Παρόμοιες συνομιλίες από το παρελθόν:{examples_text}\n\n"
        prompt += "\nΛάβε υπόψη το ιστορικό της συνομιλίας για να απαντήσεις κατάλληλα."

    messages = [{"role": "system", "content": prompt}] + context
    print(json.dumps(prompt, indent=4, ensure_ascii=False), file=log_file)
    log_file.flush()
    
    # Build the API call parameters
    api_params = {
        "model": openai_model,
        "messages": messages,
        "temperature": temperature,
        **extra_kwargs
    }
    
    # Add tools if available
    if tools:
        api_params["tools"] = tools
        api_params["tool_choice"] = "auto"

    # Use OpenAI service instead of direct client
    return openai_service.create_chat_completion(**api_params)

def retrieve_similar_conversations(query, index=None, top_k=3, intent_data=None):
    """
    Retrieve similar conversations with intent-aware filtering
    """
    # Extract primary intent
    primary_intent = intent_data.get("primary", "unknown") if intent_data else "unknown"
    
    # Generate embedding
    query_embedding = model.encode(query).tolist()
    
    # Build filter based on intent if possible
    filter_params = None
    if primary_intent != "unknown":
        filter_params = {
            "intent": {"$eq": primary_intent}
        }
    
    # Search vector database
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True,
        filter=filter_params
    )
    
    retrieved_contexts = []
    for match in results["matches"]:
        if match["score"] > 0.75:  # Only use if similarity is high enough
            retrieved_contexts.append({
                "query": match["metadata"]["query"],
                "response": match["metadata"]["response"],
                "similarity": match["score"],
                "intent": match["metadata"].get("intent", "unknown")
            })
    
    # If we didn't get enough results with the intent filter, try without it
    if len(retrieved_contexts) < 2 and filter_params:
        results = index.query(
            vector=query_embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        for match in results["matches"]:
            if match["score"] > 0.75:
                # Check if this example is already in our list
                if not any(r.get("query") == match["metadata"]["query"] for r in retrieved_contexts):
                    retrieved_contexts.append({
                        "query": match["metadata"]["query"],
                        "response": match["metadata"]["response"],
                        "similarity": match["score"],
                        "intent": match["metadata"].get("intent", "unknown")
                    })
                    if len(retrieved_contexts) >= top_k:
                        break
    
    return retrieved_contexts

def get_assistant_reply(user_id, complete_message, text_messages):
    context = get_convo_context(user_id)

    previous_assistant_message = None
    for entry in reversed(context):
        if entry.get("role") == "assistant":
            previous_assistant_message = entry.get("content")
            break

    # Classify intents with context
    intents_data = classify_intent(complete_message, previous_assistant_message=previous_assistant_message)
    print(f"Intent data for {user_id}: {json.dumps(intents_data, indent=4, ensure_ascii=False)}", file=log_file)
    log_file.flush()

    # Extract the intents list from the dictionary
    intents = intents_data.get("intents", []) if isinstance(intents_data, dict) else intents_data
    
    print(intents, file=log_file)
    try:
        response = get_openai_call_for_intent(context, intents, user_id, text_messages)
        
        # Handle function calls if present - allow for multiple rounds of function calls
        max_function_rounds = 3  # Prevent infinite loops
        current_round = 0
        
        while (response and hasattr(response.choices[0].message, 'tool_calls') and 
               response.choices[0].message.tool_calls and current_round < max_function_rounds):
            
            current_round += 1
            print(f"Function call round {current_round}", file=log_file)
            log_file.flush()
            
            # Process function calls
            tool_calls = response.choices[0].message.tool_calls
            function_results = []
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Executing function: {function_name} with args: {function_args}", file=log_file)
                log_file.flush()
                
                # Execute the calendar function
                result = execute_calendar_function(function_name, function_args)
                print(f"Function result: {json.dumps(result, ensure_ascii=False)}", file=log_file)
                log_file.flush()
                
                # Store the result
                function_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            # Add the assistant's message with tool calls to context
            context.append({
                "role": "assistant",
                "content": response.choices[0].message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]
            })
            
            # Add function results to context
            context.extend(function_results)
            
            # Make another API call to potentially make more function calls or get final response
            response = openai_service.create_chat_completion(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "system", "content": """
                    Απαντάς σε DM πελατών του 210tattoo. Χρησιμοποίησες τις λειτουργίες ημερολογίου και τώρα πρέπει να απαντήσεις στον πελάτη με βάση τα αποτελέσματα.

                    Γράφεις πάντα στα ελληνικά.
                    Πάντα στο τέλος του μηνύματος να βάζεις τα emoji \"❤️🐼\".

                    Αν το ραντεβού δημιουργήθηκε ή επιβεβαιώθηκε επιτυχώς:
                    - Επιβεβαίωσε ΜΟΝΟ την ημερομηνία και ώρα
                    - Πες ότι θα λάβει υπενθύμιση μια ώρα πριν
                    - ΜΗΝ αναφέρεις την εκτιμώμενη διάρκεια του ραντεβού
                    - ΜΗΝ αναφέρεις το κόστος που συμφωνήσατε

                    Αν υπάρχουν διαθέσιμες ώρες:
                    - Παρουσίασέ τες όμορφα και ρώτα ποια προτιμούν
                    - ΜΗΝ αναφέρεις το εύρος ημερομηνιών που έψαξες (end_date)
                    - ΜΗΝ αναφέρεις την εκτιμώμενη διάρκεια του ραντεβού
                    - ΜΗΝ αναφέρεις το κόστος που συμφωνήσατε

                    Για ακυρώσεις ραντεβού:
                    - Αν μόλις έγινε find_customer_booking και βρέθηκαν ραντεβού, προχώρησε ΑΜΕΣΩΣ να κάνεις cancel_tattoo_booking με το πρώτο/μοναδικό event_id
                    - Αν υπάρχουν πολλά ραντεβού, ακύρωσε το πιο πρόσφατο ή ρώτησε τον πελάτη ποιο θέλει
                    - Αν ακυρώθηκε επιτυχώς, επιβεβαίωσε την ακύρωση ευγενικά
                    - Αν δεν βρέθηκαν ραντεβού, ρώτησε για τον σωστό αριθμό τηλεφώνου ή ημερομηνία ραντεβού

                    Αν κάτι πήγε στραβά, ενημέρωσε ευγενικά και πρότεινε εναλλακτικές.
                """}] + context,
                temperature=1.0,
                tools=[{"type": "function", "function": func} for func in CALENDAR_FUNCTIONS],
                tool_choice="auto"
            )
        
        # Get final reply
        if response:
            reply = response.choices[0].message.content.strip() if response.choices[0].message.content else "⚠️ Προέκυψε πρόβλημα με την επεξεργασία του αιτήματός σου."
        else:
            reply = "⚠️ Προέκυψε πρόβλημα με την επεξεργασία του αιτήματός σου."
            
    except Exception as e:
        print(f"Error in get_assistant_reply: {str(e)}", file=log_file)
        log_file.flush()
        reply = "⚠️ Προέκυψε πρόβλημα με την απάντηση για ένα από τα αιτήματα σου."

    # Clean up
    redis_client.delete(f"image_analysis:{user_id}")

    return reply


system_prompt_2 = '''
Είσαι μέλος ενός στούντιο τατουάζ που χρειάζεται να εξάγει χρηστικές πληροφορίες από μια φωτογραφία τατουάζ.

1. Δώσε σύντομη περιγραφή του στυλ (fine line, ρεαλιστικό κ.λπ.), της περιοχής στο σώμα (π.χ. χέρι, μηρός) και τυχόν αξιοσημείωτων χαρακτηριστικών (σκίαση, χρώμα).
2. Υπολόγισε ΚΑΙ ανέφερε ρητά τα παρακάτω τέσσερα (4) στοιχεία σ' ΕΝΑΝ **μοναδικό** στίχο, σε αυτή τη σειρά, χωρισμένα με « | »:
   - Εκτιμώμενο ύψος σε cm (h)
   - Εκτιμώμενο πλάτος σε cm (w)
   - Ποσοστό επιφάνειας με μελάνι σε δεκαδικό (ink) π.χ. 0.45
   - Συντελεστής δυσκολίας D σύμφωνα με τον πίνακα:
       • 1.14 → απλό γραμμικό, χωρίς γέμισμα ή σκιά
       • 1.21 → λίγη απαλή σκίαση
       • 1.45 → πολύ σκιά ή ornate λεπτομέρεια
       • 1.60 → γεμισμένο μαύρο
       • 1.65 → γεμισμένο με 1 χρώμα
       • 1.85 → χρώμα + σκίαση
       • 2.10 → πολυχρωμία + έντονο shading
       • 2.50 → ρεαλιστικό (όχι πορτρέτο)
       • 3.30 → πορτρέτο / πανοπλία / υφή
       • 3.75 → πολύ μικρό script

FORMAT παραδείγματος:
«Fine line minimal house outline στον καρπό | h=5 | w=5 | ink=0.10 | D=1.14»

Μην προσθέσεις τίποτα άλλο εκτός από την περιγραφή + τη γραμμή με τις τιμές.
'''

def get_image_analysis_reply(image_path):
    """
    Analyze tattoo image using OpenAI vision API

    NOTE: Now uses OpenAIService for better error handling and retry logic
    """
    return openai_service.analyze_image(
        image_path=image_path,
        system_prompt=system_prompt_2,
        user_prompt="Ανέλυσε την εικόνα του τατουάζ.",
        temperature=OPENAI_TEMPERATURE_PRICING
    )

def schedule_processing(user_id):
    """Schedule message processing after grace period"""
    # Check if a processing task is already scheduled
    scheduled_key = f"scheduled:{user_id}"
    if redis_client.get(scheduled_key):
        # A task is already scheduled, no need to schedule another
        return
    
    # Determine a randomized grace period between GRACE_WINDOW_SECONDS+1 and +10
    random_grace = GRACE_WINDOW_SECONDS + random.randint(1, 10)

    # Mark that a task is scheduled with a buffer to avoid race conditions
    redis_client.setex(scheduled_key, random_grace + 5, "1")
    
    # Schedule the actual processing using a background task
    # For simplicity, we'll use threading here
    timer = threading.Timer(random_grace, process_user_messages, args=[user_id])
    timer.daemon = True
    timer.start()

def classify_intent(message, previous_assistant_message=None):
    """
    Classify the user's message and detect multiple intents, using previous assistant message for context.
    Returns a list of intent objects.
    """
    try:
        with open('./prompts/classification.txt', 'r', encoding='utf-8') as f:
            classification_prompt = f.read()

        # Add current date and previous assistant message to the prompt
        current_date = datetime.now().strftime('%d/%m/%Y')
        if previous_assistant_message:
            message_with_context = (
                f"[PREVIOUS_ASSISTANT]: {previous_assistant_message}\n"
                f"[CURRENT_DATE: {current_date}]\n"
                f"{message}"
            )
        else:
            message_with_context = f"[CURRENT_DATE: {current_date}]\n{message}"

        logger.info("classifying_intent", message_length=len(message))

        response = openai_service.create_chat_completion(
            model=OPENAI_MODEL_CLASSIFY,
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": message_with_context}
            ],
            temperature=OPENAI_TEMPERATURE_CLASSIFY,
            response_format={"type": "json_object"}
        )

        # Return the list of intents
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Error classifying intent: {str(e)}")
        return []


def process_user_messages(user_id):
    """Process all queued messages after grace period"""
    # Try to acquire the processing lock
    if not acquire_processing_lock(user_id):
        # Another process is already handling this user's messages
        return
    
    # Abort processing entirely if human override is active
    if is_user_muted(user_id):
        release_processing_lock(user_id)
        return
    
    try:
        # Clear the scheduled flag
        redis_client.delete(f"scheduled:{user_id}")
        
        # Wait for image analysis if pending
        pending_key = f"images_pending:{user_id}"
        pending_count = int(redis_client.get(pending_key) or "0")
        
        if pending_count > 0:
            # Reschedule processing after a short delay
            import threading
            timer = threading.Timer(3, process_user_messages, args=[user_id])
            timer.daemon = True
            timer.start()
            return

        # Get all queued messages
        messages = get_queued_messages(user_id)
        if not messages:
            # No messages to process
            return
        
        # Sort messages by timestamp
        messages.sort(key=lambda x: x["timestamp"])
        
        # Combine all message + image analyses
        combined_message = ""
        text_messages = ""
        
        image_analysis = redis_client.lrange(f"image_analysis:{user_id}", 0, -1)
        for msg in messages:
            if "message" in msg["data"] and "text" in msg["data"]["message"]:
                combined_message += msg["data"]["message"]["text"] + "\n"
                text_messages += msg["data"]["message"]["text"] + "\n"
            if "attachments" in msg["data"]["message"]:
                total_images = len(msg["data"]["message"]["attachments"])
                for i in range(total_images):
                    if msg["data"]["message"]["attachments"][i]["type"] == "image":

                        combined_message += image_analysis[i]
                        text_messages += "Ο χρήστης έστειλε μια φωτογραφία" + "\n"
        combined_message = combined_message.strip()

        # If there's no text but there is an image analysis, create a placeholder message
        # Store user's message in conversation context
        save_convo_context(user_id, {
            "role": "user",
            "content": combined_message
        })

        # Get AI reply
        bot_reply = get_assistant_reply(user_id, combined_message, text_messages)
        
        # Send the reply using messaging service (handles long messages automatically)
        responses = messaging_service.send_long_message(user_id, bot_reply)

        # Store the full reply in context
        save_convo_context(user_id, {
            "role": "assistant",
            "content": bot_reply
        })

        logger.info(
            "message_sent_to_user",
            user_id=user_id,
            message_length=len(bot_reply),
            chunks_sent=len(responses)
        )
        
        # Clear the image pending flag
        redis_client.delete(f"image_pending:{user_id}")
    
        # Clear the message queue
        clear_message_queue(user_id)
    
    finally:
        # Always release the lock
        release_processing_lock(user_id)

def acquire_processing_lock(user_id):
    """Try to acquire a lock for processing a user's messages"""
    lock_key = f"processing_lock:{user_id}"
    # Set the lock with NX (only if it doesn't exist)
    # and an expiration of 30 seconds to prevent deadlocks
    return redis_client.set(lock_key, "1", nx=True, ex=PROCESSING_LOCK_TTL)

def release_processing_lock(user_id):
    """Release the processing lock"""
    lock_key = f"processing_lock:{user_id}"
    redis_client.delete(lock_key)

def queue_user_message(user_id, message_data, has_image=False):
    """Add a message to the user's queue with a timestamp"""
    # Skip automatic processing if a human override is active
    if is_user_muted(user_id):
        return
    try:
        # Queue the message
        queue_key = f"message_queue:{user_id}"
        message_with_timestamp = {
            "timestamp": time.time(),
            "data": message_data,
            "has_image": has_image
        }
        redis_client.lpush(queue_key, json.dumps(message_with_timestamp))
        redis_client.expire(queue_key, QUEUE_TTL_SECONDS)
        
        # Schedule processing after grace period
        schedule_processing(user_id)
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error in queue_user_message: {str(e)}", file=log_file)
        log_file.flush()
        raise
    except Exception as e:
        print(f"Unexpected error in queue_user_message: {str(e)}", file=log_file)
        log_file.flush()
        raise

def get_queued_messages(user_id):
    """Get all queued messages for a user"""
    queue_key = f"message_queue:{user_id}"
    messages = redis_client.lrange(queue_key, 0, -1)
    if not messages:
        return []
    return [json.loads(msg) for msg in messages]

def clear_message_queue(user_id):
    """Clear the message queue after processing"""
    queue_key = f"message_queue:{user_id}"
    redis_client.delete(queue_key)

# ---------------------------------------------------------------------------
# Human-in-the-loop: mute automatic replies when a heart reaction is received
# ---------------------------------------------------------------------------
def mute_user(user_id, duration_seconds=MUTE_DURATION_SECONDS):
    """Mute automated replies for a user for the specified duration (default 2 h)."""
    redis_client.setex(f"mute:{user_id}", duration_seconds, "1")

def is_user_muted(user_id):
    """Check if the user is currently muted (human override active)."""
    return redis_client.exists(f"mute:{user_id}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', '3000'))
    app.run(port=port, debug=DEBUG)