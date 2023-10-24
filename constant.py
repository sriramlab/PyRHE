import os
from dotenv import load_dotenv

load_dotenv()
RESULT_DIR = os.getenv("RESULT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
