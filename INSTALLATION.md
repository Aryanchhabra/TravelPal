# TravelPal Installation Guide

This guide will walk you through the steps to set up and run TravelPal on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Step 1: Clone the Repository

```bash
git clone https://github.com/Aryanchhabra/TravelPal.git
cd TravelPal
```

## Step 2: Set Up a Virtual Environment (Recommended)

It's best to create a virtual environment to manage dependencies:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Step 4: Set Up API Key (IMPORTANT - Secure Method)

TravelPal uses the Google Gemini API for AI capabilities.

### Secure API Key Setup:

1. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Create a `.env` file in the project root directory:
   ```
   cp .env.template .env
   ```

3. Edit the `.env` file and add your API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

4. **Security Warning:** 
   - NEVER commit your `.env` file to version control
   - The `.gitignore` file is set up to exclude `.env` files
   - Do not share your API key publicly

## Step 5: Run the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Troubleshooting

- **API Key Issues**: If you see API key errors, you can also enter your key directly in the sidebar of the application. However, using the `.env` file is more secure.

- **Model Errors**: If you encounter model-specific errors, try switching to a different model in the sidebar settings.

- **Package Errors**: Ensure all dependencies are installed by running:
  ```bash
  pip install -r requirements.txt --upgrade
  ```

## Deployment

For deployment to services like Streamlit Cloud or Heroku, refer to the `DEPLOYMENT.md` file.

## Additional Configuration

You can modify the application settings in the `config.py` file. 