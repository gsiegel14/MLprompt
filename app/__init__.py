from flask import Flask, render_template
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)

# Set the secret key (for session management)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Import routes at the end to avoid circular imports
from app import main
from app import api_endpoints  # Import API endpoints