import os
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file"""
    load_dotenv()

    # Access environment variables
    tracing_enabled = os.getenv('LANGCHAIN_TRACING_V2') # for LangSmith
    endpoint = os.getenv('LANGCHAIN_ENDPOINT')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    #print(f'Tracing enabled: {tracing_enabled}')
    #print(f'LangSmith endpoint: {endpoint}')
    #print(f'OpenAI API Key: {openai_api_key}')

    return tracing_enabled, endpoint, openai_api_key

