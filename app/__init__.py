import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize extensions outside of application factory
db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    """Application factory function to create and configure the Flask app"""
    # Create the Flask app
    app = Flask(__name__)
    
    # Set the secret key (for session management)
    app.secret_key = os.environ.get("SESSION_SECRET", "atlas-prompt-optimization-platform-dev-secret-2025")
    
    # Set session cookie parameters
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
    
    # Initialize extensions with the app
    initialize_extensions(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register blueprints and routes
    register_blueprints(app)
    register_routes(app)
    
    # Import additional modules
    import_modules()
    
    return app

def initialize_extensions(app):
    """Initialize Flask extensions"""
    # Initialize SQLAlchemy
    db.init_app(app)
    
    # Log database connection info
    logger.info(f"Database configuration complete. URI: {app.config['SQLALCHEMY_DATABASE_URI'][:20]}...")
    
    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'login'  # Redirect to our login page
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User
        return User.query.get(int(user_id))
    
    # Create all tables in the database
    with app.app_context():
        db.create_all()

def register_error_handlers(app):
    """Register error handlers for the application"""
    @app.errorhandler(404)
    def page_not_found(e):
        return "Page not found", 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return "Internal server error", 500

def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    try:
        # Import and register blueprints
        from app.google_auth import google_auth
        app.register_blueprint(google_auth, url_prefix='/google_auth')
        
        # Import and register ML views blueprint
        from app.ml.views import ml_views
        app.register_blueprint(ml_views, url_prefix='/ml')
        
        # Import and register ML API blueprint
        from app.ml.routes import ml_api
        app.register_blueprint(ml_api, url_prefix='/api/ml')
        
        # Import API endpoints
        from app.api_endpoints import api_blueprint
        app.register_blueprint(api_blueprint, url_prefix='/api')
        
        logger.debug("Blueprints registered successfully")
    except Exception as e:
        logger.error(f"Error registering blueprints: {e}")
        logger.exception("Blueprint registration exception details:")

def register_routes(app):
    """Register all routes with the Flask app"""
    try:
        # Import routes
        from app.routes import index, prompts, optimize, view_optimizations, login, variables
        
        # Register routes with the Flask app
        app.add_url_rule('/', view_func=index)
        app.add_url_rule('/prompts', view_func=prompts, methods=['GET', 'POST'])
        app.add_url_rule('/optimize', view_func=optimize, methods=['GET', 'POST'])
        app.add_url_rule('/optimizations', view_func=view_optimizations)
        app.add_url_rule('/login', view_func=login, methods=['GET', 'POST'])
        app.add_url_rule('/variables', view_func=variables)
        
        logger.debug("Routes initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing routes: {e}")
        logger.exception("Route initialization exception details:")

def import_modules():
    """Import necessary modules for the application"""
    try:
        # Import models and other modules
        import app.data_module  # Make sure data_module is initialized
        import app.llm_client  # Make sure llm_client is available
        import app.workflow  # Make sure workflow module is loaded
        
        logger.debug("Modules imported successfully")
    except Exception as e:
        logger.error(f"Error importing modules: {e}")
        logger.exception("Module import exception details:")