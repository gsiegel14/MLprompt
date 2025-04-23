
"""
Flask dashboard for monitoring and administration
"""
import os
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_assets import Environment, Bundle

logger = logging.getLogger(__name__)

def create_flask_app():
    """Create the Flask dashboard app"""
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates', 'dashboard'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static', 'dashboard'))
    
    # Configure assets
    assets = Environment(app)
    
    # Register CSS and JS bundles
    css = Bundle('css/dashboard.css', 'css/charts.css', 
                 filters='cssmin', output='gen/dashboard.min.css')
    js = Bundle('js/dashboard.js', 'js/charts.js',
                filters='jsmin', output='gen/dashboard.min.js')
    
    assets.register('css_all', css)
    assets.register('js_all', js)
    
    # Routes
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('index.html', title='Prompt Optimization Dashboard')
    
    @app.route('/metrics')
    def metrics():
        """Metrics and monitoring page"""
        return render_template('metrics.html', title='Metrics')
    
    @app.route('/workflows')
    def workflows():
        """Workflow monitoring page"""
        return render_template('workflows.html', title='Workflows')
    
    @app.route('/costs')
    def costs():
        """Cost tracking page"""
        return render_template('costs.html', title='Cost Tracking')
    
    @app.route('/api/dashboard/status')
    def status():
        """API endpoint for dashboard status"""
        return jsonify({
            "status": "healthy",
            "components": {
                "api": "active",
                "database": "active",
                "prefect": "active" if os.environ.get("PREFECT_API_URL") else "inactive"
            }
        })
    
    return app
