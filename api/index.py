import sys
import os
from pathlib import Path

# Ensure project root is in Python sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import app

# Vercel ASGI Entrypoint
# Vercel's Python runtime natively detects the 'app' ASGI instance

