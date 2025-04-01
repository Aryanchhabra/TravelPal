import streamlit as st
import os
import sys

# Display a simple startup page while checking dependencies
st.set_page_config(
    page_title="TravelPal - AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

st.title("TravelPal - AI Travel Planner")
st.write("Checking dependencies and starting up...")

# Add a spinner to show loading
with st.spinner("Initializing application..."):
    try:
        # Test import the problematic dependencies one by one
        import pydantic
        st.write(f"✓ Pydantic version: {pydantic.__version__}")
        
        import langchain
        st.write(f"✓ Langchain version: {langchain.__version__}")
        
        import langchain_core
        st.write(f"✓ Langchain Core version: {langchain_core.__version__}")
        
        import google.generativeai
        st.write(f"✓ Google Generative AI library loaded")
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        st.write(f"✓ ChatGoogleGenerativeAI loaded")
        
        from langchain_community.tools import DuckDuckGoSearchRun
        st.write(f"✓ DuckDuckGoSearchRun loaded")
        
        # All dependencies are working, now we can import the main app
        st.success("All dependencies loaded successfully!")
        
        # Import the main application
        from app import main
        
        # Run the main application
        main()
    
    except Exception as e:
        st.error(f"Error starting application: {str(e)}")
        st.error(f"Type: {type(e).__name__}")
        st.code(f"Details: {str(e)}", language="python")
        
        # More debug info
        st.write("### System Information")
        st.write(f"Python version: {sys.version}")
        st.write(f"Working directory: {os.getcwd()}")
        
        # Show installed packages
        st.write("### Installed Packages")
        import pkg_resources
        installed_packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        st.code("\n".join(installed_packages), language="text") 