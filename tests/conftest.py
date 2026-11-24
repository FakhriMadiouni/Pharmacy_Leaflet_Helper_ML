import os
import sys
from pathlib import Path

# The tests always run in mock mode, whatever your .env says (real values in the environment win over .env).
os.environ["INFERENCE_MODE"] = "mock"
os.environ["EMBEDDING_MODE"] = "mock"
os.environ["GROQ_API_KEY"] = ""
os.environ["SEARCH_THRESHOLD"] = "0.20"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
