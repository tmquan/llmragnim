import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

class LLMs:
    """
    Class to hold lazy-loaded large language models (LLMs).
    """

    @property
    def nim_mixtral_llm(self):
        """Lazy-loaded property for the Nim Mixtral LLM."""
        return self._get_nim_mixtral_llm()

    @property
    def nvai_mixtral_llm(self):
        """Lazy-loaded property for the NVIDIA AI Mixtral LLM."""
        return self._get_nvai_mixtral_llm()

    @property
    def openai_gpt3_llm(self):
        """Lazy-loaded property for the OpenAI GPT-3 LLM."""
        return self._get_openai_gpt3_llm()

    def _get_nim_mixtral_llm(self):
        """Initialize and return the Nim Mixtral LLM."""
        nim_mixtral_llm = ChatOpenAI(
            model='mixtral-8x7b-instruct',
            openai_api_base='http://mixtral:9999/v1',
            openai_api_key='n/a',
            max_tokens=1024,
            model_kwargs={"frequency_penalty": -2.0}
        )
        return nim_mixtral_llm

    def _get_nvai_mixtral_llm(self):
        """Initialize and return the NVIDIA AI Mixtral LLM."""
        _ = load_dotenv('.env', override=True)
        nvidia_api_key = os.environ.get('NVIDIA_API_KEY')
        if nvidia_api_key is None:
            raise ValueError("NVIDIA_API_KEY is not set. Please set it using set_api_key('NVIDIA_API_KEY', '<your_api_key>').")
        nvai_mixtral_llm = ChatNVIDIA(model='ai-mixtral-8x7b-instruct', nvidia_api_key=nvidia_api_key)
        return nvai_mixtral_llm

    def _get_openai_gpt3_llm(self):
        """Initialize and return the OpenAI GPT-3 LLM."""
        _ = load_dotenv('.env', override=True)
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set. Please set it using set_api_key('OPENAI_API_KEY', '<your_api_key>').")
        openai_gpt3_llm = ChatOpenAI(openai_api_key=openai_api_key)
        return openai_gpt3_llm

llms = LLMs()

def set_api_key(key, value):
    """
    Set an API key in the .env file.

    Args:
        key (str): The name of the API key.
        value (str): The value of the API key.
    """
    update_env_file('.env', key, value)

def update_env_file(file_path, key, value):
    """
    Update the .env file with a new key-value pair.

    Args:
        file_path (str): Path to the .env file.
        key (str): The name of the key to update or add.
        value (str): The value of the key.
    """
    # Read the existing content of the file
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    # Check if the key already exists and update its value
    key_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f'{key}='):
            lines[i] = f'{key}={value}\n'
            key_exists = True
            break

    # If the key doesn't exist, append it to the file
    if not key_exists:
        lines.append(f'{key}={value}\n')

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)