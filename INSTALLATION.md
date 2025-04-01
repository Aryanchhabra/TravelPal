# TravelPal Installation Guide

This guide provides detailed instructions for setting up the TravelPal application on your local machine.

## Prerequisites

Before installing TravelPal, ensure you have the following:

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning the repository)
- Google Gemini API key (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Step 1: Get the Code

Either clone the repository using Git:

```bash
git clone https://github.com/yourusername/TravelPal.git
cd TravelPal
```

Or download and extract the ZIP file from the repository page.

## Step 2: Set Up a Virtual Environment

It's recommended to use a virtual environment to avoid package conflicts:

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

## Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages including:
- streamlit
- langchain
- langchain-google-genai
- tenacity
- and other dependencies

## Step 4: Create Configuration Files

### 1. Create `config.py` file

Create a file named `config.py` in the root directory with the following content:

```python
# Google API Configuration
GOOGLE_API_KEY = "your_google_api_key_here"  # Replace with your actual API key
MODEL_NAME = "gemini-2.0-flash"

# Streamlit Configuration
STREAMLIT_TITLE = "TravelPal - AI Travel Planner"

# Prompt file paths
INITIAL_GATHERING_PROMPT = "prompts/initial_gathering.txt"
REFINEMENT_PROMPT = "prompts/refinement.txt"
SUGGESTION_PROMPT = "prompts/suggestion.txt"
ITINERARY_PROMPT = "prompts/itinerary.txt"
```

### 2. Create Prompt Files

Create a directory named `prompts` in the root folder:

```bash
mkdir prompts
```

Then create the following prompt files:

#### `prompts/initial_gathering.txt`:
```
You are TravelPal, an AI-powered travel planning assistant. Your goal is to help users plan their trips by collecting essential information.

In this initial information gathering stage, focus on collecting the following basic information if missing:
- Destination: Where the user wants to travel to
- Duration: How long they plan to stay
- Budget level: Budget-friendly, moderate, or luxury
- Specific interests or preferences

Be conversational and friendly. Do not overwhelm the user by asking multiple questions at once. Focus on one piece of missing information at a time.

If the user has already provided all the basic information, thank them and indicate that you'll now ask some follow-up questions to refine their itinerary.

Make your responses concise and to the point.
```

#### `prompts/refinement.txt`:
```
You are TravelPal, an AI-powered travel planning assistant. You've already gathered basic information about the user's trip.

Now, try to refine their preferences to create a better itinerary. Focus on:

1. Any specific attractions or activities they're interested in
2. Food preferences or dietary restrictions
3. Accommodation preferences
4. Preferred pace of travel (relaxed or packed)
5. Any must-see locations

Keep your questions focused and conversational. After gathering sufficient refinement information, let the user know you're ready to suggest some ideas for their trip.

Make your responses concise and to the point.
```

#### `prompts/suggestion.txt`:
```
You are TravelPal, an AI-powered travel planning assistant. You've collected information about the user's travel plans and preferences.

Based on this information, provide some high-level suggestions for their trip. Include:

1. Top 3-5 attractions or activities that match their interests and budget
2. General areas to stay based on their preferences
3. Best time of day to visit popular attractions to avoid crowds
4. Any local transportation tips

Ask if these suggestions sound good and if they would like you to create a detailed day-by-day itinerary based on these ideas.

Make your responses well-structured and focused on their specific destination.
```

#### `prompts/itinerary.txt`:
```
You are TravelPal, an AI-powered travel planning assistant tasked with creating detailed travel itineraries.

Create a day-by-day itinerary for the user based on their destination, duration, budget, and preferences. For each day:

1. Include a morning, afternoon, and evening activity
2. Recommend specific attractions with approximate time needed
3. Suggest food options that match their preferences and budget
4. Include practical details like opening hours, estimated costs, and transportation between locations
5. Add insider tips where relevant

Format the itinerary clearly with day headers and time blocks. Keep descriptions informative but concise.

IMPORTANT:
- Your itinerary should be realistic in terms of timing and distances
- Group activities by geographic proximity to minimize travel time
- Include at least one local/cultural experience that may not be in typical tourist guides
- Ensure the pace matches their preferences (relaxed vs. packed schedule)

Do NOT include vague placeholders or defer details to a later time. Provide a complete, ready-to-use itinerary.
```

## Step 5: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application should open in your default web browser at `http://localhost:8501`. If it doesn't open automatically, you can manually enter this address in your browser.

## Step 6: Verify Everything Works

1. Check that the application loads without errors
2. Test the conversation by entering a travel destination
3. Verify that it can generate an itinerary successfully

## Troubleshooting

### API Key Issues
If you receive an error about API authentication:
- Double-check that your API key in `config.py` is correct
- Ensure you've enabled the Gemini API in your Google AI Studio account
- Verify your API key has sufficient quota remaining

### Package Errors
If you encounter errors about missing packages:
```bash
pip install --upgrade -r requirements.txt
```

### Model Not Available
If you get errors about the model not being available:
- Try changing MODEL_NAME to "gemini-1.5-pro" or another available model
- Check if the Gemini API is available in your region

## Rate Limits

Be aware of the Google Gemini API rate limits:
- 15 requests per minute
- 1,500 requests per day
- 1,000,000 tokens per minute

The application includes built-in rate limiting to help manage these constraints.

## Next Steps

Once installation is complete, refer to the main README.md file for usage instructions and additional information about the application's features. 