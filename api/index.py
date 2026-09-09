import os
import sys

# Ensure the project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set a writable temp dir for uploads before importing the app
os.environ.setdefault("UPLOAD_FOLDER", "/tmp/uploads")

from main import app
