# TravelPal Configuration File
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#=================================================
# API CONFIGURATION
#=================================================
# Replace with your actual Google Gemini API key from Google AI Studio
# Get your API key at: https://makersuite.google.com/app/apikey
# IMPORTANT: Without a valid API key, the application will fall back to simplified functionality
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key_here")

# Model settings - change only if you have access to different models
MODEL_NAME = "gemini-1.5-flash"  # More reliable than gemini-2.0-flash in many cases
# Alternative models: "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"

# Fallback model if primary model fails
FALLBACK_MODEL_NAME = "gemini-1.5-flash"
USE_SAFETY_SETTINGS = False  # Set to False if having issues with API

#=================================================
# APPLICATION SETTINGS
#=================================================
# Application title displayed in browser tab
STREAMLIT_TITLE = "TravelPal - AI Travel Planner"

#=================================================
# PROMPT FILE PATHS
#=================================================
# System prompts for different conversation stages
INITIAL_GATHERING_PROMPT = "prompts/initial_gathering.txt"
REFINEMENT_PROMPT = "prompts/refinement.txt"
SUGGESTION_PROMPT = "prompts/suggestion.txt"
ITINERARY_PROMPT = "prompts/itinerary.txt"

#=================================================
# ADVANCED MODEL SETTINGS (optional)
#=================================================
# Modify these if you need to fine-tune the model output
MODEL_TEMPERATURE = 0.7        # Higher = more creative, Lower = more deterministic
MODEL_TOP_P = 0.9              # Controls diversity of generated text
MODEL_TOP_K = 32               # Limits vocabulary choices
MODEL_MAX_OUTPUT_TOKENS = 1024 # Maximum response length

#=================================================
# RATE LIMITING SETTINGS
#=================================================
# Settings for API rate limiting (per Gemini models limits)
RPM_LIMIT = 15                # Requests per minute
RPD_LIMIT = 1500              # Requests per day
TPM_LIMIT = 1000000           # Tokens per minute
ESTIMATED_TOKENS_PER_REQUEST = 100  # Estimated tokens per typical request

#=================================================
# ERROR HANDLING
#=================================================
# Control how the application handles errors
MAX_RETRIES = 3               # Maximum number of retry attempts for API calls
ENABLE_MOCK_RESPONSES = True  # Generate mock responses when API calls fail
DEBUG_MODE = False            # Print detailed error information 