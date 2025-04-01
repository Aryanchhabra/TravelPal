import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import json
from typing import Dict, List, Optional, Any
import config
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta
import threading
from collections import deque
import re
from dataclasses import dataclass, field
import argparse
import sys
import concurrent.futures
from langchain.prompts import MessagesPlaceholder

# Add command line arguments for better control
def parse_args():
    parser = argparse.ArgumentParser(description='TravelPal AI Travel Planner')
    parser.add_argument('--disable-safety', action='store_true', help='Disable safety settings')
    parser.add_argument('--model', type=str, help='Override model name (e.g., gemini-1.5-flash)')
    return parser.parse_args()

# Get command line arguments early
args = parse_args()
if args.model:
    config.MODEL_NAME = args.model
    print(f"Model overridden to: {config.MODEL_NAME}")

# Rate limiting controls
class RateLimiter:
    def __init__(self):
        self.requests_per_min = deque(maxlen=15)  # 15 RPM limit
        self.requests_per_day = deque(maxlen=1500)  # 1,500 RPD limit
        self.tokens_per_min = 0  # Track token usage per minute
        self.last_token_reset = datetime.now()
        self.lock = threading.Lock()

    def check_rate_limits(self, estimated_tokens: int = 100) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        with self.lock:
            # Reset token counter if minute has passed
            if (now - self.last_token_reset).total_seconds() >= 60:
                self.tokens_per_min = 0
                self.last_token_reset = now

            # Clean up old timestamps
            while self.requests_per_min and self.requests_per_min[0] < minute_ago:
                self.requests_per_min.popleft()
            while self.requests_per_day and self.requests_per_day[0] < day_ago:
                self.requests_per_day.popleft()
            
            # Check limits
            if len(self.requests_per_min) >= 15:  # 15 RPM
                return False
            if len(self.requests_per_day) >= 1500:  # 1,500 RPD
                return False
            if self.tokens_per_min + estimated_tokens > 1_000_000:  # 1M TPM
                return False
            
            # Add new timestamp and update token count
            self.requests_per_min.append(now)
            self.requests_per_day.append(now)
            self.tokens_per_min += estimated_tokens
            return True

rate_limiter = RateLimiter()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
def get_llm():
    """Initialize the LLM with improved error handling"""
    try:
        # Print API key status (without revealing the key)
        if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your_google_api_key_here":
            print("Warning: No valid API key provided")
            st.sidebar.error("⚠️ Please enter your Google Gemini API key in the sidebar")
            raise ValueError("No valid API key provided")
            
        # Create the model with proper error handling
        model_kwargs = {
            "model": config.MODEL_NAME,
            "temperature": config.MODEL_TEMPERATURE,
            "google_api_key": config.GOOGLE_API_KEY,
            "convert_system_message_to_human": True,
            "max_output_tokens": config.MODEL_MAX_OUTPUT_TOKENS,
            "top_p": config.MODEL_TOP_P,
            "top_k": config.MODEL_TOP_K,
        }
        
        # Only add safety settings if configured to use them
        if getattr(config, 'USE_SAFETY_SETTINGS', False):
            try:
                model_kwargs["safety_settings"] = {
                    "HARASSMENT": "block_none", 
                    "HATE": "block_none", 
                    "SEXUAL": "block_none", 
                    "DANGEROUS": "block_none"
                }
            except Exception as safety_error:
                print(f"Warning: Safety settings not applied: {str(safety_error)}")
                
        return ChatGoogleGenerativeAI(**model_kwargs)
    except Exception as e:
        error_msg = str(e)
        print(f"Error initializing LLM: {error_msg}")
        
        # If the error is about an invalid API key, provide a helpful message
        if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
            st.sidebar.error("⚠️ Invalid API Key. Please check your Google Gemini API key.")
            print("Authentication error - invalid API key")
        
        # Try with fallback model
        fallback_model = getattr(config, 'FALLBACK_MODEL_NAME', "gemini-1.5-flash")
        if config.MODEL_NAME != fallback_model:
            print(f"Attempting with fallback model: {fallback_model}")
            try:
                fallback_kwargs = {
                    "model": fallback_model,
                    "temperature": 0.7,
                    "google_api_key": config.GOOGLE_API_KEY,
                    "convert_system_message_to_human": True,
                    "max_output_tokens": 1024,
                    "top_p": 0.9,
                    "top_k": 40
                }
                
                # Simplified safety settings for fallback
                return ChatGoogleGenerativeAI(**fallback_kwargs)
            except Exception as fallback_error:
                print(f"Fallback model also failed: {str(fallback_error)}")
        
        # If we get here, both attempts failed
        raise

def run_with_rate_limit(func, *args, estimated_tokens: int = 100, **kwargs):
    """Run a function with rate limiting"""
    if not rate_limiter.check_rate_limits(estimated_tokens):
        wait_time = 60  # Wait for 1 minute if rate limited
        st.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        if not rate_limiter.check_rate_limits(estimated_tokens):
            raise Exception("Still rate limited after waiting. Please try again later.")
    return func(*args, **kwargs)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
def run_chain(chain: LLMChain, input_text: str) -> str:
    """Run a chain with rate limiting and retry logic"""
    try:
        return run_with_rate_limit(chain.run, input=input_text)
    except Exception as e:
        if "429" in str(e):
            st.warning("Rate limit reached. Taking a short break before retrying...")
            time.sleep(20)  # Longer sleep on rate limit
        raise

def generate_mock_itinerary(destination):
    """Generate a mock itinerary as a fallback when API fails."""
    days = 3  # Default days
    
    # Enriched fallback data
    city_attractions = {
        "Paris": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", "Montmartre", "Seine River Cruise", 
                 "Arc de Triomphe", "Champs-Élysées", "Musée d'Orsay", "Luxembourg Gardens", "Sacré-Cœur Basilica"],
        "London": ["Buckingham Palace", "Tower of London", "British Museum", "London Eye", "Westminster Abbey", 
                  "Hyde Park", "Tate Modern", "Covent Garden", "Natural History Museum", "Camden Market"],
        "New York": ["Statue of Liberty", "Central Park", "Empire State Building", "Times Square", "Brooklyn Bridge", 
                    "Metropolitan Museum of Art", "Broadway Show", "High Line", "One World Observatory", "Fifth Avenue"],
        "Tokyo": ["Tokyo Skytree", "Senso-ji Temple", "Meiji Shrine", "Shibuya Crossing", "Shinjuku Gyoen", 
                 "Tokyo Disneyland", "Akihabara", "Tsukiji Fish Market", "Imperial Palace", "Harajuku"],
        "Rome": ["Colosseum", "Vatican City", "Trevi Fountain", "Roman Forum", "Pantheon", 
                "Spanish Steps", "Borghese Gallery", "Piazza Navona", "Palatine Hill", "Trastevere"],
    }
    
    restaurants = {
        "Paris": ["Le Jules Verne", "Café de Flore", "L'Ambroisie", "Au Pied de Cochon", "Septime"],
        "London": ["The Ledbury", "Dishoom", "Gordon Ramsay Restaurant", "Duck & Waffle", "The Ivy"],
        "New York": ["Katz's Delicatessen", "Eleven Madison Park", "Peter Luger", "Gramercy Tavern", "Le Bernardin"],
        "Tokyo": ["Sukiyabashi Jiro", "Sushi Saito", "Narisawa", "Den", "Sushi Dai"],
        "Rome": ["La Pergola", "Roscioli", "Armando al Pantheon", "Pierluigi", "Da Enzo al 29"],
    }
    
    # Get attractions for the specified destination or use generic ones
    attractions = city_attractions.get(destination, 
        ["Local Museum", "Historic District", "Main Square", "National Park", "Famous Landmark", 
         "Cultural Center", "Local Market", "Scenic Viewpoint", "Historical Monument", "Shopping District"])
    
    local_food = restaurants.get(destination,
        ["Local Restaurant", "Traditional Café", "Famous Bakery", "Street Food Market", "Fine Dining Experience"])
    
    # Build a more detailed itinerary
    itinerary = [f"# {destination} Travel Itinerary\n\n"]
    itinerary.append("*This is a fallback itinerary generated when the AI service cannot be reached. For a personalized itinerary, please try again later.*\n\n")
    
    import random
    random.shuffle(attractions)  # Randomize attractions order
    
    for day in range(1, min(days + 1, 5)):  # Cap at 5 days for fallback
        itinerary.append(f"## Day {day}")
        
        # Morning
        morning_attraction = attractions[day % len(attractions)]
        itinerary.append(f"### Morning")
        itinerary.append(f"- Start your day with breakfast at a local café")
        itinerary.append(f"- Visit {morning_attraction}")
        itinerary.append(f"- Explore the surrounding area and take photos")
        
        # Lunch
        lunch_spot = local_food[day % len(local_food)]
        itinerary.append(f"\n### Afternoon")
        itinerary.append(f"- Enjoy lunch at {lunch_spot}")
        itinerary.append(f"- Visit {attractions[(day + 1) % len(attractions)]}")
        itinerary.append(f"- Take a break at a local café or park")
        
        # Evening
        evening_attraction = attractions[(day + 2) % len(attractions)]
        itinerary.append(f"\n### Evening")
        itinerary.append(f"- Have dinner at {local_food[(day + 1) % len(local_food)]}")
        itinerary.append(f"- Experience {evening_attraction} or the nightlife")
        itinerary.append("\n")
    
    # Add some practical tips
    itinerary.append("## Practical Tips")
    itinerary.append("- Best time to visit: Check local weather forecasts before planning your day")
    itinerary.append("- Local transportation: Research public transportation options")
    itinerary.append("- Accommodations: Look for hotels or rentals in central areas")
    itinerary.append("- Currency: Check if you need to exchange to local currency")
    itinerary.append("\n*For a more personalized itinerary with current information, please try connecting again with your API key*")
    
    return "\n".join(itinerary)

def generate_fallback_response(st, user_msg, current_stage="initial"):
    """Generate a fallback response when the API call fails"""
    # Extract destination if possible
    destination_match = re.search(r"(?:to|in|for|visit|at)\s+([A-Za-z\s]+)(?:\.|\?|$|,|\s+for|\s+in|\s+on)", user_msg)
    destination = destination_match.group(1).strip() if destination_match else "your destination"
    
    if current_stage == "itinerary":
        return generate_mock_itinerary(destination)
    
    st.warning("⚠️ Unable to connect to the AI service. Providing a helpful fallback response.")
    
    if "itinerary" in user_msg.lower() or "plan" in user_msg.lower():
        return generate_mock_itinerary(destination)
    
    if "budget" in user_msg.lower() or "cost" in user_msg.lower():
        return f"To plan a trip to {destination} on a budget, consider:\n\n" + \
               "- Traveling during off-peak seasons\n" + \
               "- Looking for budget accommodations like hostels or vacation rentals\n" + \
               "- Using public transportation\n" + \
               "- Eating at local, less touristy restaurants\n" + \
               "- Taking advantage of free attractions and activities\n\n" + \
               "For a more personalized budget plan, please try connecting again with your API key."
    
    if "hotel" in user_msg.lower() or "stay" in user_msg.lower() or "accommodation" in user_msg.lower():
        return f"For accommodations in {destination}, you might want to consider:\n\n" + \
               "- City center hotels for convenience\n" + \
               "- Boutique hotels for unique experiences\n" + \
               "- Vacation rentals for longer stays\n" + \
               "- Hostels for budget travelers\n" + \
               "- Luxury hotels for premium experiences\n\n" + \
               "For specific recommendations, please try connecting again with your API key."
    
    # Generic fallback
    return f"I understand you're interested in traveling to {destination}. " + \
           "I'd be happy to help with your travel planning once the connection to the AI service is restored. " + \
           "In the meantime, you might want to consider researching:\n\n" + \
           "- Best time to visit based on weather and crowds\n" + \
           "- Top attractions and activities\n" + \
           "- Local transportation options\n" + \
           "- Accommodation choices in different neighborhoods\n" + \
           "- Local cuisine and dining recommendations\n\n" + \
           "Please check your API key and try again for personalized assistance."

def create_chain(prompt_file):
    """Create a chain with error handling and fallback mechanisms"""
    try:
        # Load prompt from file
        try:
            prompt_content = load_prompt(prompt_file)
        except Exception as e:
            print(f"Error loading prompt from {prompt_file}: {str(e)}")
            # Use a simple fallback prompt if file loading fails
            prompt_content = "You are a helpful AI travel assistant. Provide detailed and useful information."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_content),
            ("human", "{input}")
        ])
        
        # Initialize LLM if not already done
        if 'llm' not in st.session_state:
            try:
                st.session_state.llm = get_llm()
            except Exception as e:
                print(f"Error initializing LLM in create_chain: {str(e)}")
                return None
                
        return LLMChain(llm=st.session_state.llm, prompt=prompt)
    except Exception as e:
        print(f"Error creating chain: {str(e)}")
        return None

def load_prompt(file_path: str) -> str:
    """Load prompt from file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading prompt from {file_path}: {str(e)}")
        # Fallback prompt if file can't be loaded
        return "You are a helpful AI travel assistant. Provide detailed and useful information."

def generate_itinerary() -> str:
    """Generate itinerary with better error handling and detailed fallbacks"""
    try:
        basic_info = st.session_state.user_info['basic_info']
        
        # Auto-fill missing information if we have a destination
        if basic_info['destination']:
            if not basic_info['duration']:
                basic_info['duration'] = '5 days'  # Default duration
            if not basic_info['budget']:
                basic_info['budget'] = 'Moderate'  # Default budget
            if not basic_info['preferences']:
                # Add some default preferences based on destination
                destination_defaults = {
                    'Switzerland': ['mountains', 'scenic', 'hiking', 'nature'],
                    'France': ['culture', 'food', 'art', 'architecture'],
                    'Italy': ['history', 'food', 'art', 'architecture'],
                    'Japan': ['culture', 'food', 'technology', 'history'],
                    'USA': ['shopping', 'entertainment', 'sightseeing'],
                    'Australia': ['beach', 'nature', 'wildlife', 'adventure'],
                    'UK': ['history', 'culture', 'museums', 'architecture'],
                    'Spain': ['culture', 'beaches', 'food', 'architecture'],
                    'Germany': ['history', 'beer', 'architecture', 'culture'],
                    'Thailand': ['beaches', 'temples', 'food', 'nightlife'],
                    'Greece': ['history', 'islands', 'beaches', 'food'],
                    'India': ['culture', 'history', 'food', 'spirituality']
                }
                
                for dest, prefs in destination_defaults.items():
                    if dest.lower() in basic_info['destination'].lower():
                        basic_info['preferences'].extend(prefs)
                        break
                
                # If no specific match, add general preferences
                if not basic_info['preferences']:
                    basic_info['preferences'] = ['sightseeing', 'culture', 'food', 'nature']
        
        # First try with the search info
        try:
            search_results = run_with_rate_limit(
                search_travel_info,
                f"top tourist attractions and travel tips for {basic_info['destination']}"
            )
        except Exception as e:
            print(f"Search failed: {str(e)}")
            search_results = f"Unable to search for specific details about {basic_info['destination']} at this time."
        
        try:
            # Check if there's a valid API key before continuing
            if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your_google_api_key_here":
                st.sidebar.error("⚠️ Please enter your Google Gemini API key in the sidebar")
                return generate_mock_itinerary(
                    basic_info['destination']
                )
            
            # Initialize the LLM if needed
            if 'llm' not in st.session_state:
                try:
                    st.session_state.llm = get_llm()
                except Exception as e:
                    print(f"Error initializing LLM in generate_itinerary: {str(e)}")
                    return generate_mock_itinerary(basic_info['destination'])
            
            # Create the chain and call the API
            prompt_content = load_prompt(config.ITINERARY_PROMPT)
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_content),
                ("human", "{input}")
            ])
            
            # Create the chain with proper error handling
            try:
                chain = LLMChain(llm=st.session_state.llm, prompt=prompt)
            except Exception as chain_error:
                print(f"Error creating chain: {str(chain_error)}")
                return generate_mock_itinerary(basic_info['destination'])
                
            user_info_str = json.dumps({
                "destination": basic_info['destination'],
                "duration": basic_info['duration'] or '5 days',
                "budget": basic_info['budget'] or 'Moderate',
                "preferences": basic_info['preferences']
            }, indent=2)
            
            context = f"""
User Information:
{user_info_str}

Search Results (use these for inspiration and specific details):
{search_results}

Instructions:
1. Create a day-by-day itinerary for {basic_info['destination']} for {basic_info['duration'] or '5 days'}
2. Include specific attractions, activities, and restaurants for each day
3. Add practical information like opening hours and transportation
4. Make this a COMPLETE, DETAILED, READY-TO-USE itinerary
5. Focus on activities matching these preferences: {', '.join(basic_info['preferences']) if basic_info['preferences'] else 'sightseeing, culture, food, nature'}
"""
            return run_chain(chain, context)
        except Exception as e:
            print(f"Itinerary generation failed: {str(e)}")
            # If API calls fail, use an enhanced mock itinerary as fallback
            return generate_mock_itinerary(
                basic_info['destination']
            )
    except Exception as e:
        error_detail = str(e)
        print(f"Error in itinerary generation: {error_detail}")
        if "API key" in error_detail.lower() or "authentication" in error_detail.lower():
            st.sidebar.error("⚠️ Please check your Google Gemini API key")
            return "I need a valid API key to generate a detailed itinerary. Please add your key in the sidebar."
        elif "429" in error_detail:
            return "I apologize, but I've reached the rate limit. Please wait a minute before requesting an itinerary."
        else:
            return f"I'm having trouble creating your itinerary at the moment. Please try again in a few moments. Technical details: {error_detail[:100]}..."

def update_user_info(message: str):
    """Update user info based on the message content."""
    message_lower = message.lower()
    basic_info = st.session_state.user_info['basic_info']
    
    # Extract location information - expanded list
    locations = {
        'destination': ['zurich', 'switzerland', 'geneva', 'bern', 'lucerne', 'basel', 'interlaken', 'zermatt', 
                        'paris', 'france', 'london', 'uk', 'rome', 'italy', 'berlin', 'germany', 'barcelona', 'spain',
                        'amsterdam', 'netherlands', 'vienna', 'austria', 'prague', 'czech', 'tokyo', 'japan', 
                        'new york', 'usa', 'sydney', 'australia', 'dubai', 'uae', 'toronto', 'canada', 
                        'singapore', 'bangkok', 'thailand', 'istanbul', 'turkey', 'athens', 'greece',
                        'stockholm', 'sweden', 'lisbon', 'portugal', 'dublin', 'ireland', 'zurich', 'switzerland'],
        'starting_location': ['pune', 'india', 'mumbai', 'delhi', 'bangalore', 'hyderabad', 'new york', 'london', 
                              'paris', 'berlin', 'rome', 'tokyo', 'sydney', 'dubai', 'toronto', 'los angeles', 
                              'chicago', 'miami', 'bangkok', 'singapore']
    }
    
    # More efficient destination extraction using regex
    location_pattern = r"(?:visit|go\s+to|travel\s+to|going\s+to|seeing|explore)\s+([A-Za-z\s]+)(?:for|in|during|this|next|\.|\?|$)"
    if match := re.search(location_pattern, message_lower):
        potential_destination = match.group(1).strip().lower()
        # Check if extracted location is in our list
        for loc in locations['destination']:
            if loc in potential_destination:
                basic_info['destination'] = loc.title()
                break
    else:
        # Fallback to simpler matching
        for location_type, keywords in locations.items():
            for keyword in keywords:
                if keyword in message_lower:
                    basic_info[location_type] = keyword.title()
                    break

    # Extract dates and duration - enhanced pattern matching
    duration_patterns = [
        r'(\d+)\s*days?',
        r'(\d+)\s*weeks?',
        r'for\s*(\d+)\s*days?',
        r'about\s*(\d+)\s*days?',
        r'(\d+)-day',
        r'stay(?:ing)?\s*for\s*(\d+)',
        r'spending\s*(\d+)\s*days'
    ]
    
    for pattern in duration_patterns:
        if match := re.search(pattern, message_lower):
            days = int(match.group(1))
            if 'week' in pattern:
                days *= 7
            basic_info['duration'] = f"{days} days"
            break

    # Extract budget information - enhanced
    budget_patterns = {
        'Budget': [r'budget(?:[-\s]friendly)?', 'cheap', 'affordable', 'low cost', 'economical', 'inexpensive'],
        'Moderate': [r'moderate', 'mid(?:[-\s]range)', 'average', 'reasonable', 'normal', 'standard'],
        'Luxury': [r'luxury', 'high(?:[-\s]end)', 'expensive', 'premium', 'deluxe', 'upscale']
    }
    
    for budget_type, patterns in budget_patterns.items():
        if any(re.search(pattern, message_lower) for pattern in patterns):
            basic_info['budget'] = budget_type
            break

    # Extract preferences more aggressively with advanced pattern matching
    preference_patterns = [
        (r'(?:interested in|enjoy|love|like|want to see|want to do|prefer)\s+([^,.!?]+)', 1),
        (r'fan of\s+([^,.!?]+)', 1),
        (r'my interests? (?:include|are|is)\s+([^,.!?]+)', 1)
    ]
    
    for pattern, group in preference_patterns:
        for match in re.finditer(pattern, message_lower):
            preferences_text = match.group(group).strip()
            # Split by common separators
            for pref in re.split(r',|\sand\s|&', preferences_text):
                pref = pref.strip()
                if pref and len(pref) > 2 and pref not in basic_info['preferences']:
                    basic_info['preferences'].append(pref)
    
    # Also check for direct mentions of common preferences
    preference_keywords = [
        'hiking', 'trekking', 'nature', 'mountains', 'lakes', 'scenic', 'landscape',
        'culture', 'history', 'museum', 'art', 'architecture',
        'food', 'cuisine', 'restaurant', 'dining', 'gastronomy',
        'shopping', 'markets', 'stores', 'mall',
        'relaxation', 'spa', 'wellness', 'beach',
        'adventure', 'sports', 'activities', 'nightlife',
        'photography', 'sightseeing', 'landmarks', 'local'
    ]
    
    for keyword in preference_keywords:
        if keyword in message_lower and keyword not in basic_info['preferences']:
            basic_info['preferences'].append(keyword)

    # If we have a destination but no preferences, add some default ones based on the destination
    if basic_info['destination'] and not basic_info['preferences']:
        destination_defaults = {
            'Switzerland': ['mountains', 'scenic', 'hiking', 'nature'],
            'France': ['culture', 'food', 'art', 'architecture'],
            'Italy': ['history', 'food', 'art', 'architecture'],
            'Japan': ['culture', 'food', 'technology', 'history'],
            'USA': ['shopping', 'entertainment', 'sightseeing'],
            'Australia': ['beach', 'nature', 'wildlife', 'adventure'],
            'UK': ['history', 'culture', 'museums', 'architecture'],
            'Spain': ['culture', 'beaches', 'food', 'architecture'],
            'Germany': ['history', 'beer', 'architecture', 'culture'],
            'Thailand': ['beaches', 'temples', 'food', 'nightlife'],
            'Greece': ['history', 'islands', 'beaches', 'food'],
            'India': ['culture', 'history', 'food', 'spirituality']
        }
        
        for dest, prefs in destination_defaults.items():
            if dest.lower() in basic_info['destination'].lower():
                basic_info['preferences'].extend(prefs)
                break
        
        # If no specific match, add general preferences based on destination name
        if not basic_info['preferences'] and basic_info['destination']:
            dest_lower = basic_info['destination'].lower()
            if any(word in dest_lower for word in ['beach', 'island', 'coast']):
                basic_info['preferences'] = ['beaches', 'relaxation', 'water activities', 'seafood']
            elif any(word in dest_lower for word in ['mountain', 'alps', 'hill']):
                basic_info['preferences'] = ['hiking', 'nature', 'mountains', 'scenic views']
            elif any(word in dest_lower for word in ['city', 'capital', 'metro']):
                basic_info['preferences'] = ['sightseeing', 'culture', 'food', 'architecture']

    # Auto-fill missing information with defaults if we have a destination
    if basic_info['destination'] and not basic_info['duration']:
        basic_info['duration'] = '5 days'  # Default duration
        
    if basic_info['destination'] and not basic_info['budget']:
        basic_info['budget'] = 'Moderate'  # Default budget

def main():
    # Set up the page with a modern dark theme and custom styling
    st.set_page_config(
        page_title=config.STREAMLIT_TITLE,
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .stButton button {
            background-color: #4F46E5;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #3730A3;
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .stProgress > div > div {
            background-color: #4F46E5;
        }
        .stExpander {
            background-color: #1A1F29;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .stChat {
            border-radius: 15px;
            margin: 0.5rem 0;
        }
        .user-info {
            background-color: #1A1F29;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .logo-text {
            background: linear-gradient(45deg, #4F46E5, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .header-card {
            background: linear-gradient(135deg, #1A1F29, #111827);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .card {
            background-color: #1A1F29;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        .chat-input {
            background-color: #1A1F29;
            border-radius: 10px;
            border: 1px solid #2D3748;
            color: white;
            padding: 1rem;
            margin-top: 1rem;
        }
        footer {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with logo and title
    st.markdown("""
        <div class="header-card">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 2.5rem; margin-right: 1rem;">✈️</div>
                <div>
                    <h1 class="logo-text" style="margin: 0;">TravelPal</h1>
                    <p style="margin: 0; color: #A0AEC0;">Your AI travel planning companion</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with API status
    with st.sidebar:
        st.markdown("""
            <style>
            .sidebar-header {
                font-size: 1.5rem;
                font-weight: bold;
                margin-bottom: 1rem;
                color: #FFFFFF;
            }
            .help-container {
                background-color: #1A1F29;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Quick start guide
        st.markdown('<p class="sidebar-header">🚀 Quick Start</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="help-container">
                <p>Just tell TravelPal your destination to get started!</p>
                <p>Examples:</p>
                <ul>
                    <li><i>"I want to visit Paris for 4 days"</i></li>
                    <li><i>"Plan a trip to Tokyo"</i></li>
                    <li><i>"I'm thinking about a hiking trip in Switzerland"</i></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # About section
        st.markdown('<p class="sidebar-header">ℹ️ About</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="help-container">
                <p>TravelPal creates personalized travel itineraries using AI.</p>
                <p>Simply tell it where you want to go, and it will generate a day-by-day plan tailored to your preferences.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # API Key Configuration
        st.markdown('<p class="sidebar-header">🔑 API Setup</p>', unsafe_allow_html=True)
        api_key = st.text_input("Google API Key", value=config.GOOGLE_API_KEY if config.GOOGLE_API_KEY != "your_google_api_key_here" else "", type="password")
        if api_key and api_key != config.GOOGLE_API_KEY:
            config.GOOGLE_API_KEY = api_key
            st.success("API key updated! Please refresh the messages to use the new key.")
    
    # Main content area
    try:
        initialize_session_state()
        
        if not st.session_state.messages:
            # Welcome message
            welcome_message = """Hi there! 👋 I'm TravelPal, your AI travel planning assistant. 

Just tell me where you'd like to go, and I'll create a personalized itinerary for you.

For example, you can say: "I want to visit Rome for 5 days" or "Plan a trip to Tokyo"
"""
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    except Exception as e:
        if "429" in str(e):
            st.error("API rate limit reached. Please wait a few minutes before trying again.")
            st.stop()
        raise

    # Current Travel Information - only shown once destination is known
    if st.session_state.user_info['basic_info']['destination']:
        with st.expander("Trip Information", expanded=False):
            st.markdown('<div class="user-info">', unsafe_allow_html=True)
            
            basic_info = st.session_state.user_info['basic_info']
            
            # Format as a travel card
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div class="card">
                        <h3 style="margin-top: 0;">✈️ Trip Summary</h3>
                        <p><strong>Destination:</strong> {basic_info['destination'] or 'Not specified'}</p>
                        <p><strong>Duration:</strong> {basic_info['duration'] or 'Not specified'}</p>
                        <p><strong>Budget:</strong> {basic_info['budget'] or 'Not specified'}</p>
                    </div>
                    
                    <div class="card">
                        <h3 style="margin-top: 0;">❤️ Preferences</h3>
                        <p>{', '.join(basic_info['preferences']) if basic_info['preferences'] else 'No preferences specified'}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Where would you like to travel?", key="chat_input"):
        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append(HumanMessage(content=prompt))
        
        with st.chat_message("assistant"):
            with st.spinner("Creating your travel plan..."):
                try:
                    # Update user info based on message
                    update_user_info(prompt)
                    
                    # Determine if we should generate itinerary now
                    generate_now = False
                    
                    # If we have a destination, check conditions for immediate generation
                    if st.session_state.user_info['basic_info']['destination']:
                        # Keywords that indicate user wants an itinerary
                        itinerary_keywords = ['itinerary', 'plan', 'schedule', 'create', 'generate', 'make', 'suggest']
                        
                        # Generate immediately if:
                        # 1. User explicitly asks for an itinerary
                        if any(word in prompt.lower() for word in itinerary_keywords):
                            generate_now = True
                            
                        # 2. This is their first message with a destination (first impression matters)
                        if len(st.session_state.messages) <= 3:
                            generate_now = True
                            
                        # 3. They've provided comprehensive info in one go
                        if len(prompt.split()) > 15 and any(word in prompt.lower() for word in ['visit', 'travel', 'trip', 'go to']):
                            generate_now = True
                            
                        # 4. We've already exchanged several messages (don't drag on)
                        if len(st.session_state.messages) > 5:
                            generate_now = True
                        
                        # Process based on decision
                        if generate_now:
                            with st.spinner("Creating your personalized travel itinerary..."):
                                itinerary = generate_itinerary()
                                st.session_state.messages.append({"role": "assistant", "content": itinerary})
                                st.session_state.conversation_history.append(AIMessage(content=itinerary))
                                st.write(itinerary)
                                st.session_state.current_stage = 'itinerary'
                        else:
                            # Ask ONE focused question before generating
                            response = f"Great choice! I'll plan your trip to {st.session_state.user_info['basic_info']['destination']}. "
                            
                            # Identify ONLY the most important missing information
                            missing = []
                            basic_info = st.session_state.user_info['basic_info']
                            
                            if not basic_info['duration']:
                                missing.append("how long you'll be staying")
                            elif not basic_info['preferences']:
                                missing.append("what activities or sights you're most interested in")
                            
                            # Ask at most ONE question
                            if missing and len(missing) == 1:
                                response += f"Could you tell me {missing[0]}? Or I can create an itinerary now with what I know."
                            else:
                                # Generate anyway if we have multiple missing items - don't overwhelm
                                with st.spinner("Creating your personalized travel itinerary..."):
                                    itinerary = generate_itinerary()
                                    st.session_state.messages.append({"role": "assistant", "content": itinerary})
                                    st.session_state.conversation_history.append(AIMessage(content=itinerary))
                                    st.write(itinerary)
                                    st.session_state.current_stage = 'itinerary'
                                return
                            
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.session_state.conversation_history.append(AIMessage(content=response))
                            st.write(response)
                    else:
                        # No destination yet, ask for one
                        response = """I'd be happy to plan your trip! To get started, could you tell me where you'd like to travel to?"""
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.conversation_history.append(AIMessage(content=response))
                        st.write(response)
                
                except Exception as e:
                    error_message = str(e)
                    if "429" in error_message:
                        st.error("I'm experiencing high demand right now. Please try again in a few seconds.")
                    else:
                        st.error(f"An error occurred: {error_message}")
                        # Log the error for debugging
                        print(f"Error in processing: {error_message}")

# Initialize LLM
try:
    llm = get_llm()
except Exception as e:
    st.error(f"Error initializing AI model: {str(e)}")
    st.stop()

# Initialize search tool
search = DuckDuckGoSearchRun()

def search_travel_info(query: str) -> str:
    """Search for travel-related information."""
    try:
        search_query = f"travel guide {query}"
        results = search.run(search_query)
        # Truncate results to avoid token limits
        return results[:6000] if results else "No search results found."
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return "Unable to retrieve search results at this time."

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'initial'
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {
            'basic_info': {
                'destination': None,
                'starting_location': None,
                'dates': None,
                'duration': None,
                'budget': None,
                'purpose': None,
                'group_size': None,
                'preferences': []
            },
            'refined_info': {
                'dietary': {
                    'restrictions': None,
                    'preferences': None,
                    'allergies': None
                },
                'mobility': {
                    'walking_tolerance': None,
                    'pace_preference': None,
                    'tour_preference': None
                },
                'accommodation': {
                    'type': None,
                    'location': None,
                    'amenities': [],
                    'rating': None
                },
                'interests': {
                    'cultural': [],
                    'outdoor': [],
                    'shopping': [],
                    'nightlife': []
                },
                'special_requirements': {
                    'rest_periods': None,
                    'timing_preference': None,
                    'must_see': []
                }
            }
        }
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def check_basic_info_complete() -> bool:
    """Check if we have enough information to generate an itinerary."""
    basic_info = st.session_state.user_info['basic_info']
    # Only require destination - everything else can be defaulted
    return basic_info['destination'] is not None

def check_refined_info_complete() -> bool:
    """Check if refined information is sufficiently collected."""
    # Simplified check - don't require too much refined information
    basic_info = st.session_state.user_info['basic_info']
    return len(basic_info.get('preferences', [])) > 0

def get_missing_info() -> List[str]:
    """Get a list of missing required information."""
    missing = []
    basic_info = st.session_state.user_info['basic_info']
    
    if not basic_info['destination']:
        missing.append("destination")
    if not basic_info['duration']:
        missing.append("trip duration")
    if not basic_info['budget']:
        missing.append("budget range")
    if not basic_info['preferences']:
        missing.append("travel interests")
        
    return missing

def determine_next_question(missing_info: List[str]) -> str:
    """Determine the next most appropriate question to ask."""
    if not missing_info:
        return None
        
    question_templates = {
        "destination": "Where would you like to travel to?",
        "trip duration": "How long would you like to stay?",
        "budget range": "What's your budget range for this trip?",
        "travel interests": "What kind of activities or sights are you interested in?"
    }
    
    return question_templates.get(missing_info[0])

def format_conversation_history() -> str:
    """Format the conversation history for context."""
    formatted_history = "Previous conversation:\n"
    
    # Get the last 5 message pairs for context (to save tokens)
    recent_messages = st.session_state.conversation_history[-10:]
    
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            formatted_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted_history += f"Assistant: {msg.content[:200]}...\n" if len(msg.content) > 200 else f"Assistant: {msg.content}\n"
            
    # Add current user information
    formatted_history += "\nCurrent user information:\n"
    basic_info = st.session_state.user_info['basic_info']
    for key, value in basic_info.items():
        if value:
            formatted_history += f"{key.replace('_', ' ').title()}: {value}\n"
            
    return formatted_history

def process_message(user_input):
    """Process user message with improved error handling and fallbacks"""
    if not user_input.strip():
        return "I didn't catch that. Please tell me where you'd like to travel."
    
    # Update user info
    update_user_info(user_input)
    
    # Track usage for rate limiting
    st.session_state.rate_limiter.track_request()
    try:
        token_estimate = estimate_tokens(user_input)
        st.session_state.rate_limiter.track_tokens(token_estimate)
    except Exception as e:
        print(f"Error tracking tokens: {str(e)}")
    
    try:
        # Get the current conversation stage
        current_stage = st.session_state.current_stage
        
        # Initialize LLM if not already done
        if 'llm' not in st.session_state:
            try:
                st.session_state.llm = get_llm()
            except Exception as e:
                error_msg = str(e)
                print(f"LLM initialization error: {error_msg}")
                st.error(f"⚠️ Error initializing AI model: {error_msg[:100]}...")
                
                # Return a fallback response based on the user's message
                return generate_fallback_response(st, user_input, current_stage)
        
        # Get the appropriate prompt file path for the current stage
        prompt_file = getattr(config, f"{current_stage.upper()}_GATHERING_PROMPT", config.INITIAL_GATHERING_PROMPT)
        
        # Create the chain with proper error handling
        chain = create_chain(prompt_file)
        if chain is None:
            return generate_fallback_response(st, user_input, current_stage)
        
        # Run the chain with timeout and retry logic
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    chain.run, 
                    input=user_input
                )
                
                # Set a timeout for the API call
                try:
                    response = future.result(timeout=20)  # 20 second timeout
                except concurrent.futures.TimeoutError:
                    print("API call timed out after 20 seconds")
                    return generate_fallback_response(st, user_input, current_stage)
            
            # Log successful response and update conversation stages if needed
            print(f"Got response for stage: {current_stage}")
            
            # Transition conversation stages based on completion criteria
            try:
                if current_stage == "initial" and check_basic_info_complete():
                    st.session_state.current_stage = "refining"
                    print("Transitioned to refining stage")
                elif current_stage == "refining" and check_refined_info_complete():
                    st.session_state.current_stage = "itinerary"
                    print("Transitioned to itinerary stage")
            except Exception as stage_error:
                print(f"Error updating conversation stage: {str(stage_error)}")
            
            # Add function call for itinerary generation if needed
            if current_stage == "itinerary" and "itinerary" in user_input.lower():
                try:
                    # Add itinerary to response
                    response += "\n\n" + generate_itinerary()
                except Exception as itinerary_error:
                    print(f"Error generating itinerary: {str(itinerary_error)}")
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"Chain execution error: {error_msg}")
            
            # Check for specific errors
            if "rate limit" in error_msg.lower():
                return "I apologize, but we've hit the rate limit for our AI service. Please try again in a moment."
            elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                st.error("⚠️ API Key Error: Please check your Google Gemini API key in the sidebar.")
                return "There seems to be an issue with the API key. Please check that you've entered a valid Google Gemini API key in the sidebar."
            
            # Use fallback for general errors
            return generate_fallback_response(st, user_input, current_stage)
            
    except Exception as outer_e:
        print(f"Unexpected error in process_message: {str(outer_e)}")
        return generate_fallback_response(st, user_input, current_stage)

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a piece of text."""
    # Simple estimation: 1 token ~= 4 characters in English
    return len(text) // 4

if __name__ == "__main__":
    main() 