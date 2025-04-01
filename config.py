# TravelPal Configuration File
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file (local development)
load_dotenv()

# Check if running on Streamlit Cloud - if so, use secrets
def get_api_key():
    """
    Securely get API key without exposing it.
    Do not log or display this key anywhere in the application.
    """
    # First try Streamlit secrets (for cloud deployment)
    try:
        key = st.secrets["GOOGLE_API_KEY"]
        if key and key != "your_google_api_key_here":
            # Return a masked indicator instead of the actual key
            return key
    except Exception:
        pass
        
    # Fall back to environment variable (for local development)
    key = os.getenv("GOOGLE_API_KEY")
    if key and key != "your_google_api_key_here":
        return key
        
    # Return placeholder if no valid key found
    return "your_google_api_key_here"

#=================================================
# API CONFIGURATION
#=================================================
# Get API key - checks Streamlit secrets first, then .env file
GOOGLE_API_KEY = get_api_key()

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
# Much more conservative values for cloud deployment
MODEL_TEMPERATURE = 0.5
MODEL_TOP_P = 0.8
MODEL_TOP_K = 30
MODEL_MAX_OUTPUT_TOKENS = 800

#=================================================
# RATE LIMITING SETTINGS
#=================================================
# Settings for API rate limiting (per Gemini models limits)
RPM_LIMIT = 5                 # Requests per minute (very conservative)
RPD_LIMIT = 500               # Requests per day
TPM_LIMIT = 250000            # Tokens per minute
ESTIMATED_TOKENS_PER_REQUEST = 100