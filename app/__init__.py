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
app.secret_key = os.environ.get("SESSION_SECRET", "atlas-prompt-optimization-platform-dev-secret-2025")
# Set session cookie parameters - adjusted for development environment
app.config['SESSION_COOKIE_SECURE'] = False  # Allow cookies over HTTP in development
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookies
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session timeout in seconds (1 hour)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True  # Refresh session on each request

# Configure database with improved connection pool settings
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,  # Maximum number of connections to keep
    'pool_timeout': 30,  # Seconds to wait before timing out
    'pool_recycle': 1800,  # Recycle connections after 30 minutes
    'max_overflow': 15,  # Maximum number of connections to create beyond pool_size
    'pool_pre_ping': True,  # Enable connection testing before use to avoid stale connections
    'connect_args': {
        'connect_timeout': 10,  # Connection timeout in seconds
        'keepalives': 1,  # Enable keepalive packets
        'keepalives_idle': 30,  # Seconds between TCP keepalive packets
        'keepalives_interval': 10,  # Seconds between keepalive probes
        'keepalives_count': 5  # Maximum number of keepalive packets before dropping connection
    }
}

# Initialize SQLAlchemy with more robust connection handling
db = SQLAlchemy(app)

# Log database connection info
logger.info(f"Database configuration complete. URI: {app.config['SQLALCHEMY_DATABASE_URI'][:20]}...")

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect to our login page
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    # Use our new error-handling method
    return User.get_by_id(user_id)

# Create all tables in the database
with app.app_context():
    db.create_all()

# Import and register blueprints
from app.google_auth import google_auth
app.register_blueprint(google_auth, url_prefix='/google_auth')

# Import and register ML views blueprint
from app.ml.views import ml_views
app.register_blueprint(ml_views, url_prefix='/ml')

# Import and register ML API blueprint
from app.ml.routes import ml_api
app.register_blueprint(ml_api, url_prefix='/api/ml')

# Import routes at the end to avoid circular imports
from app import main
from app import api_endpoints  # Import API endpoints