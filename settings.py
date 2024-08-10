from dotenv import load_dotenv
from os import environ, path

load_dotenv(verbose=True)

dotenv_path = path.join(path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

AUDIO_DIR = environ.get("AUDIO_DIR")
HIRAGANA_MODEL_PATH = environ.get("HIRAGANA_MODEL_PATH")
