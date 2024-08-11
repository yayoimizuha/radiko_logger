from dotenv import load_dotenv
from os import environ, path

load_dotenv(verbose=True)

dotenv_path = path.join(path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

AUDIO_DIR = environ.get("AUDIO_DIR")
HIRAGANA_MODEL_PATH = environ.get("HIRAGANA_MODEL_PATH")
HIRAGANA_MODEL_INSTRUCTION = environ.get("HIRAGANA_MODEL_INSTRUCTION")
VLLM_MODEL_PATH = environ.get("VLLM_MODEL_PATH")
GEMINI_API_KEY = environ.get("GEMINI_API_KEY")
