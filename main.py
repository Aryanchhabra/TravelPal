import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Simple configuration
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL = "gemini-1.5-flash"

def main():
    st.set_page_config(
        page_title="TravelPal - AI Travel Planner",
        page_icon="✈️",
        layout="wide"
    )
    
    # Header
    st.title("✈️ TravelPal")
    st.subheader("Your AI Travel Planning Assistant")
    
    # API Key input
    api_key = st.sidebar.text_input("Google Gemini API Key", value=API_KEY, type="password")
    
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar")
        st.stop()
    
    try:
        # Try to initialize the model as a simple test
        model = ChatGoogleGenerativeAI(
            model=MODEL,
            temperature=0.7,
            google_api_key=api_key
        )
        
        st.success("✅ API connection successful! You can now use the main app.")
        st.markdown("Run the main app with: `streamlit run app.py`")
        
    except Exception as e:
        st.error(f"❌ Error initializing AI model: {str(e)}")
        st.info("Recommendations to fix this error:")
        st.markdown("""
        1. Make sure your API key is correct and has access to the Gemini models
        2. Try refreshing the page
        3. Check if you've reached the API rate limits
        4. Make sure you have a stable internet connection
        """)

if __name__ == "__main__":
    main() 