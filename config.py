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

# Model settings - use the most reliable model
MODEL_NAME = "gemini-1.5-flash"

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
# SIMPLIFIED MODEL SETTINGS
#=================================================
# Conservative values that are less likely to cause errors
MODEL_TEMPERATURE = 0.5
MODEL_TOP_P = 0.8
MODEL_TOP_K = 30
MODEL_MAX_OUTPUT_TOKENS = 800

#=================================================
# RATE LIMITING SETTINGS
#=================================================
# Settings for API rate limiting (per Gemini models limits)
RPM_LIMIT = 10                # Requests per minute (reduced)
RPD_LIMIT = 1000              # Requests per day
TPM_LIMIT = 500000            # Tokens per minute
ESTIMATED_TOKENS_PER_REQUEST = 100 