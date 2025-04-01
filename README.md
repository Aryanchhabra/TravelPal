# TravelPal: AI-Powered Travel Planning Assistant

## 📋 Overview

TravelPal is an innovative AI-powered travel planning assistant that creates personalized travel itineraries with minimal user input. Using Google's Gemini 2.0 Flash API, TravelPal can understand natural language requests and generate detailed day-by-day itineraries complete with attractions, activities, dining options, and practical travel tips.

Unlike traditional travel planners that require extensive information gathering, TravelPal can generate itineraries with just a destination, intelligently filling in missing details and adapting to user preferences. The application features a beautiful, responsive user interface designed for ease of use and visual appeal.

## ✨ Key Features

- **One-Shot Itinerary Generation**: Create complete travel itineraries by simply mentioning a destination
- **Advanced NLP Understanding**: Smart extraction of travel details using regex pattern matching
- **AI-Powered Recommendations**: Destination-specific suggestions based on intelligent defaults
- **Beautiful Dark Mode UI**: Modern interface with animated cards and intuitive layout
- **Smart Conversation Flow**: Minimal questions with maximum information extraction
- **Real-time Travel Information**: Integration with up-to-date travel data
- **Error-Resistant Design**: Robust error handling and fallbacks at every level
- **Optimized Performance**: Efficient code with token management and response truncation
- **Responsive UX Design**: Adapts perfectly to different screen sizes
- **Intelligent Default System**: Auto-fills missing information based on destination type

## 🔐 Security and API Keys

**IMPORTANT: API Key Security**

TravelPal requires a Google Gemini API key to function properly:

- **NEVER commit your API key to version control**
- Use the `.env` file method described in the installation guide for secure key storage
- The application provides a secure password-masked field for entering your API key in the UI
- The `.gitignore` file is configured to exclude sensitive files (`.env`, credentials)

For detailed setup instructions, see the [Installation Guide](INSTALLATION.md).

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Quick Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Aryanchhabra/TravelPal.git
   cd TravelPal
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

5. Start the application:
   ```bash
   streamlit run app.py
   ```

For complete installation details, see the [Installation Guide](INSTALLATION.md).

## 📱 Usage Guide

### Getting Started

1. When you first open TravelPal, you'll be greeted with a welcome message
2. Simply type your destination in the chat input, for example:
   - "I want to visit Paris for 4 days"
   - "Plan a trip to Tokyo"
   - "I'm thinking of going to Switzerland"

### Providing Details (Optional)

TravelPal only needs a destination to create an itinerary, but you can provide additional details if you wish:
- Duration: "I want to stay for 7 days"
- Budget: "I'm looking for a budget-friendly trip" or "I want a luxury experience"
- Preferences: "I'm interested in hiking, museums, and local cuisine"

### Generating an Itinerary

Once you've provided your destination, TravelPal will automatically generate a personalized itinerary. If it needs any additional information, it will ask just one simple question before proceeding.

### Viewing and Managing Your Trip

- Your trip information is displayed in the collapsible "Trip Information" panel
- The itinerary will appear in the chat interface
- You can ask follow-up questions about your itinerary or request modifications

## 🧠 Technical Architecture

TravelPal is built with the following technology stack:

- **Frontend & Backend**: Streamlit for the web interface and application logic
- **LLM Integration**: LangChain for structured interactions with Google's Gemini API
- **AI Model**: Google Gemini 2.0 Flash for natural language understanding and generation
- **Information Retrieval**: DuckDuckGo search tool for real-time travel information
- **Pattern Recognition**: Advanced regex-based information extraction

### Component Breakdown

1. **User Interface (Streamlit)**
   - Responsive dark-themed UI with gradient accents
   - Card-based information display with hover animations
   - Chat interface with optimized response rendering
   - Collapsible information panels for clean presentation

2. **Natural Language Processing**
   - Sophisticated regex pattern matching for information extraction
   - Smart default preferences based on destination type
   - Context-aware conversation flow with minimal questions
   - Optimized token usage with response truncation

3. **Itinerary Generation System**
   - One-shot generation with minimal required information
   - Integration of search results with user preferences
   - Contextual templating with destination-specific instructions
   - Automatic budget and duration inference when missing

4. **Error Handling**
   - Comprehensive try-except blocks throughout the codebase
   - Graceful fallbacks for API failures and rate limits
   - Informative error messages for troubleshooting
   - Automatic recovery from common failure points

## 📂 Project Structure

```
TravelPal/
├── app.py                  # Main application file
├── config.py               # Configuration and API keys
├── .env                    # Environment variables (not in version control)
├── .env.template           # Template for environment variables
├── .gitignore              # Git ignore configuration
├── prompts/                # Prompt templates
│   ├── initial_gathering.txt
│   ├── refinement.txt
│   ├── suggestion.txt
│   └── itinerary.txt
├── requirements.txt        # Dependencies
├── INSTALLATION.md         # Detailed installation guide
└── README.md               # Documentation
```

## 🛠️ Future Improvements

- Multi-destination trip planning
- Integration with hotel and flight booking APIs
- PDF export of itineraries
- User accounts for saving multiple trip plans
- Map visualization of itineraries
- Social sharing features
- Mobile application version

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Google Gemini API for powering the AI capabilities
- Streamlit for the web application framework
- LangChain for simplified LLM interactions
- All open-source libraries used in this project
