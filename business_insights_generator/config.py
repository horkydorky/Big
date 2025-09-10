import os
from dotenv import load_dotenv

def get_google_api_key():
    """
    Loads the Google API key from the .env file.

    Raises:
        ValueError: If the GOOGLE_API_KEY is not found in the environment variables.

    Returns:
        str: The Google API key.
    """
    # Load environment variables from a .env file if it exists
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please create a .env file and add your key.")
        
    return api_key