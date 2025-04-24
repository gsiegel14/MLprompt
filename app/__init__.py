from flask import Flask, render_template
import os
import logging
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)

# Set the secret key (for session management)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'google_auth.login'  # Redirect to Google login

@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    return User.query.get(int(user_id))

# Create all tables in the database
with app.app_context():
    db.create_all()

# Import and register blueprints
from app.google_auth import google_auth
app.register_blueprint(google_auth, url_prefix='/google_auth')

# Import routes at the end to avoid circular imports
from app import main
from app import api_endpoints  # Import API endpoints