# TravelPal: AI-Powered Travel Planning Assistant

![TravelPal Logo](https://via.placeholder.com/800x400?text=TravelPal)

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

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TravelPal.git
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

4. Create a `config.py` file with your API credentials:
   ```python
   # config.py
   GOOGLE_API_KEY = "your_google_api_key"
   MODEL_NAME = "gemini-2.0-flash"
   STREAMLIT_TITLE = "TravelPal - AI Travel Planner"
   
   # Prompt file paths
   INITIAL_GATHERING_PROMPT = "prompts/initial_gathering.txt"
   REFINEMENT_PROMPT = "prompts/refinement.txt"
   SUGGESTION_PROMPT = "prompts/suggestion.txt"
   ITINERARY_PROMPT = "prompts/itinerary.txt"
   ```

5. Create the prompt files in a `prompts` folder (examples provided in the repository)

## 🏃‍♂️ Running the App

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

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
├── prompts/                # Prompt templates
│   ├── initial_gathering.txt
│   ├── refinement.txt
│   ├── suggestion.txt
│   └── itinerary.txt
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

### Key Functions

- `update_user_info()`: Advanced pattern-based information extraction
- `generate_itinerary()`: Creates travel itineraries with smart defaults
- `search_travel_info()`: Retrieves and optimizes travel information
- `format_conversation_history()`: Token-efficient conversation tracking

## 🛠️ Future Improvements

- Multi-destination trip planning
- Integration with hotel and flight booking APIs
- PDF export of itineraries
- User accounts for saving multiple trip plans
- Map visualization of itineraries
- Social sharing features
- Mobile application version

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Google Gemini API for powering the AI capabilities
- Streamlit for the web application framework
- LangChain for simplified LLM interactions

---

<p align="center">
  <b>Created by [Your Name]</b><br>
  <a href="https://github.com/yourusername">GitHub</a> •
  <a href="https://linkedin.com/in/yourusername">LinkedIn</a>
</p>

## Security and API Keys

🔐 **IMPORTANT: API Key Security**

TravelPal requires a Google Gemini API key to function properly:

- **NEVER commit your API key to version control**
- Use the `.env` file method described in the installation guide for secure key storage
- The application provides a secure password-masked field for entering your API key in the UI
- The `.gitignore` file is configured to exclude sensitive files (`.env`, credentials)

For detailed setup instructions, see the [Installation Guide](INSTALLATION.md).

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API key** (securely in a `.env` file):
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for the powerful language model
- Streamlit for the web application framework
- All open-source libraries used in this project 