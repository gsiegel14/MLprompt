
"""
Flask dashboard for monitoring and admin interface
"""
import os
import json
from datetime import datetime, timedelta
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_assets import Environment, Bundle
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from src.app.config import settings

logger = logging.getLogger(__name__)

def create_flask_app():
    """Create the Flask dashboard app"""
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    # Configure assets
    assets = Environment(app)
    
    # Register CSS and JS bundles
    css = Bundle('css/dashboard.css', 'css/charts.css', 
                 filters='cssmin', output='gen/dashboard.min.css')
    js = Bundle('js/dashboard.js', 'js/charts.js',
                filters='jsmin', output='gen/dashboard.min.js')
    
    assets.register('css_all', css)
    assets.register('js_all', js)
    
    # Route: Dashboard home
    @app.route('/')
    def index():
        """Main dashboard page"""
        system_status = {
            "api": "active",
            "database": "active" if hasattr(settings, "DATABASE_URL") else "inactive",
            "prefect": "active" if settings.PREFECT_ENABLED else "inactive",
            "llm_service": "active",
            "cache": "active" if settings.LLM_CACHE_ENABLED else "inactive"
        }
        
        # Get recent experiments (mock data for now)
        recent_experiments = get_recent_experiments()
        
        # Get token usage data (mock data for now)
        token_usage = get_token_usage_data()
        
        # Get performance metrics (mock data for now)
        performance_metrics = get_performance_metrics()
        
        return render_template(
            'dashboard/index.html', 
            title='Prompt Optimization Dashboard',
            system_status=system_status,
            recent_experiments=recent_experiments,
            token_usage=token_usage,
            performance_metrics=performance_metrics
        )
    
    # Route: Experiments
    @app.route('/experiments')
    def experiments():
        """Experiments monitoring page"""
        experiments_data = get_experiments_data()
        
        return render_template(
            'dashboard/experiments.html', 
            title='Experiments',
            experiments=experiments_data
        )
    
    # Route: Experiment detail
    @app.route('/experiments/<experiment_id>')
    def experiment_detail(experiment_id):
        """Experiment detail page"""
        experiment = get_experiment_by_id(experiment_id)
        
        if not experiment:
            return render_template('dashboard/error.html', 
                                  message=f"Experiment {experiment_id} not found"), 404
        
        # Get visualization data
        metrics_chart = create_metrics_chart(experiment)
        
        return render_template(
            'dashboard/experiment_detail.html',
            title=f"Experiment {experiment_id}",
            experiment=experiment,
            metrics_chart=metrics_chart
        )
    
    # Route: Costs
    @app.route('/costs')
    def costs():
        """Cost tracking page"""
        # Get date range from query params or default to last 30 days
        days = request.args.get('days', 30, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        cost_data = get_cost_data(start_date)
        cost_chart = create_cost_chart(cost_data)
        
        return render_template(
            'dashboard/costs.html', 
            title='Cost Tracking',
            cost_data=cost_data,
            cost_chart=cost_chart,
            days=days
        )
    
    # Route: Workflows
    @app.route('/workflows')
    def workflows():
        """Workflow monitoring page"""
        if not settings.PREFECT_ENABLED:
            message = "Prefect is not enabled. Configure PREFECT_ENABLED in settings to enable workflow monitoring."
            return render_template('dashboard/error.html', message=message)
        
        workflows_data = get_workflows_data()
        
        return render_template(
            'dashboard/workflows.html', 
            title='Workflows',
            workflows=workflows_data
        )
    
    # Route: Prompts
    @app.route('/prompts')
    def prompts():
        """Prompts comparison page"""
        prompts_data = get_prompts_data()
        
        return render_template(
            'dashboard/prompts.html',
            title='Prompt Comparison',
            prompts=prompts_data
        )
    
    # API: System status
    @app.route('/api/status')
    def api_status():
        """API endpoint for system status"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": "active",
                "database": "active" if hasattr(settings, "DATABASE_URL") else "inactive",
                "prefect": "active" if settings.PREFECT_ENABLED else "inactive",
                "llm_service": "active",
                "cache": "active" if settings.LLM_CACHE_ENABLED else "inactive"
            }
        })
    
    # API: Get recent experiments
    @app.route('/api/experiments/recent')
    def api_recent_experiments():
        """API endpoint for recent experiments"""
        return jsonify(get_recent_experiments())
    
    # API: Get token usage
    @app.route('/api/costs/tokens')
    def api_token_usage():
        """API endpoint for token usage"""
        return jsonify(get_token_usage_data())
    
    # Create necessary static directories
    os.makedirs(os.path.join(app.static_folder, 'gen'), exist_ok=True)
    os.makedirs(os.path.join(app.static_folder, 'css'), exist_ok=True)
    os.makedirs(os.path.join(app.static_folder, 'js'), exist_ok=True)
    
    return app

# Helper functions for dashboard data

def get_recent_experiments(limit=5):
    """Get recent experiments data (mock data for now)"""
    # In a production environment, this would fetch from a database
    experiments_dir = os.path.join(settings.DATA_DIR, "../experiments")
    
    experiments = []
    try:
        # Get subdirectories in experiments directory
        experiment_dirs = [d for d in os.listdir(experiments_dir) 
                          if os.path.isdir(os.path.join(experiments_dir, d))]
        
        # Sort by name (timestamp) in descending order
        experiment_dirs.sort(reverse=True)
        
        # Get data for most recent experiments
        for exp_dir in experiment_dirs[:limit]:
            # Try to parse metrics
            metrics_files = []
            exp_path = os.path.join(experiments_dir, exp_dir)
            
            # Look for metrics files
            for f in os.listdir(exp_path):
                if f.startswith('metrics_') and f.endswith('.json'):
                    metrics_files.append(f)
            
            # Sort metrics files
            metrics_files.sort()
            
            # Get final metrics if available
            final_metrics = {}
            if metrics_files:
                final_metrics_file = os.path.join(exp_path, metrics_files[-1])
                try:
                    with open(final_metrics_file, 'r') as f:
                        final_metrics = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading metrics file {final_metrics_file}: {e}")
            
            # Create experiment data
            experiment = {
                "id": exp_dir,
                "timestamp": exp_dir,
                "iterations": len(metrics_files),
                "final_score": final_metrics.get('overall_score', 0),
                "status": "completed" if metrics_files else "in_progress"
            }
            
            experiments.append(experiment)
            
    except Exception as e:
        logger.error(f"Error getting recent experiments: {e}")
    
    return experiments

def get_token_usage_data():
    """Get token usage data (mock data for now)"""
    # In a production environment, this would fetch from a database or Redis
    return {
        "total_prompt_tokens": 125000,
        "total_completion_tokens": 78500,
        "estimated_cost": 2.86,
        "models": {
            "gemini-2.5-flash": {
                "prompt_tokens": 95000,
                "completion_tokens": 58500,
                "cost": 1.86
            },
            "gemini-2.5-pro": {
                "prompt_tokens": 30000,
                "completion_tokens": 20000,
                "cost": 1.00
            }
        }
    }

def get_performance_metrics():
    """Get performance metrics (mock data for now)"""
    return {
        "api_latency_ms": 125,
        "llm_latency_ms": 850,
        "cache_hit_rate": 0.68,
        "success_rate": 0.995
    }

def get_experiments_data():
    """Get all experiments data (mock data for now)"""
    return get_recent_experiments(limit=100)

def get_experiment_by_id(experiment_id):
    """Get experiment by ID (mock data for now)"""
    experiments_dir = os.path.join(settings.DATA_DIR, "../experiments")
    exp_dir = os.path.join(experiments_dir, experiment_id)
    
    if not os.path.exists(exp_dir):
        return None
    
    # Initialize experiment data
    experiment = {
        "id": experiment_id,
        "timestamp": experiment_id,
        "metrics": [],
        "prompts": [],
        "examples": []
    }
    
    # Load metrics
    metrics_files = [f for f in os.listdir(exp_dir) 
                    if f.startswith('metrics_') and f.endswith('.json')]
    metrics_files.sort()
    
    for metrics_file in metrics_files:
        metrics_path = os.path.join(exp_dir, metrics_file)
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                experiment["metrics"].append(metrics)
        except Exception as e:
            logger.error(f"Error loading metrics file {metrics_path}: {e}")
    
    # Load prompts
    prompts_files = [f for f in os.listdir(exp_dir) 
                    if f.startswith('prompts_') and f.endswith('.json')]
    prompts_files.sort()
    
    for prompts_file in prompts_files:
        prompts_path = os.path.join(exp_dir, prompts_file)
        try:
            with open(prompts_path, 'r') as f:
                prompts = json.load(f)
                experiment["prompts"].append(prompts)
        except Exception as e:
            logger.error(f"Error loading prompts file {prompts_path}: {e}")
    
    # Add reasoning if available
    reasoning_files = [f for f in os.listdir(exp_dir) 
                      if f.startswith('reasoning_') and f.endswith('.txt')]
    reasoning_files.sort()
    
    experiment["reasoning"] = []
    for reasoning_file in reasoning_files:
        reasoning_path = os.path.join(exp_dir, reasoning_file)
        try:
            with open(reasoning_path, 'r') as f:
                reasoning = f.read()
                experiment["reasoning"].append(reasoning)
        except Exception as e:
            logger.error(f"Error loading reasoning file {reasoning_path}: {e}")
    
    return experiment

def get_cost_data(start_date):
    """Get cost data (mock data for now)"""
    # Generate daily data from start_date until today
    today = datetime.now()
    dates = []
    current_date = start_date
    
    while current_date <= today:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    # Generate random cost data
    import random
    
    data = []
    cumulative_cost = 0
    
    for date in dates:
        daily_cost = round(random.uniform(0.05, 0.2), 2)
        cumulative_cost += daily_cost
        
        data.append({
            "date": date,
            "daily_cost": daily_cost,
            "cumulative_cost": round(cumulative_cost, 2),
            "prompt_tokens": int(random.uniform(3000, 8000)),
            "completion_tokens": int(random.uniform(2000, 5000))
        })
    
    return data

def get_workflows_data():
    """Get workflows data (mock data for now)"""
    return [
        {
            "id": "flow-20250423-001",
            "name": "prompt_optimization",
            "status": "completed",
            "started": "2025-04-23T10:15:30",
            "completed": "2025-04-23T10:25:45",
            "duration": "10m 15s"
        },
        {
            "id": "flow-20250422-002",
            "name": "prompt_optimization",
            "status": "completed",
            "started": "2025-04-22T15:30:20",
            "completed": "2025-04-22T15:45:12",
            "duration": "14m 52s"
        },
        {
            "id": "flow-20250422-001",
            "name": "dataset_processing",
            "status": "failed",
            "started": "2025-04-22T12:05:10",
            "completed": "2025-04-22T12:08:32",
            "duration": "3m 22s",
            "error": "ValidationError: Dataset format invalid"
        }
    ]

def get_prompts_data():
    """Get prompts comparison data (mock data for now)"""
    # In a real implementation, this would load from a database or files
    prompts_dir = os.path.join(settings.DATA_DIR, "../prompts")
    
    system_prompts = []
    output_prompts = []
    
    # Load system prompts
    system_dir = os.path.join(prompts_dir, "system")
    if os.path.exists(system_dir):
        for file in os.listdir(system_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(system_dir, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        system_prompts.append({
                            "name": file,
                            "content": content,
                            "created": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
                except Exception as e:
                    logger.error(f"Error loading system prompt file {file_path}: {e}")
    
    # Load output prompts
    output_dir = os.path.join(prompts_dir, "output")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(output_dir, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        output_prompts.append({
                            "name": file,
                            "content": content,
                            "created": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
                except Exception as e:
                    logger.error(f"Error loading output prompt file {file_path}: {e}")
    
    return {
        "system_prompts": system_prompts,
        "output_prompts": output_prompts
    }

def create_metrics_chart(experiment):
    """Create metrics chart for experiment visualization"""
    # Extract iteration numbers and scores
    iterations = []
    scores = []
    
    for i, metrics in enumerate(experiment.get("metrics", [])):
        iterations.append(i + 1)
        scores.append(metrics.get("overall_score", 0))
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations, 
        y=scores,
        mode='lines+markers',
        name='Overall Score'
    ))
    
    fig.update_layout(
        title="Score Progression",
        xaxis_title="Iteration",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1])
    )
    
    # Convert to JSON for embedding in template
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_cost_chart(cost_data):
    """Create cost chart visualization"""
    # Extract dates and costs
    dates = [d["date"] for d in cost_data]
    daily_costs = [d["daily_cost"] for d in cost_data]
    cumulative_costs = [d["cumulative_cost"] for d in cost_data]
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add daily cost bars
    fig.add_trace(go.Bar(
        x=dates,
        y=daily_costs,
        name='Daily Cost',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Add cumulative cost line
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_costs,
        mode='lines+markers',
        name='Cumulative Cost',
        marker_color='rgb(26, 118, 255)',
        yaxis='y2'
    ))
    
    # Update layout for dual y-axis
    fig.update_layout(
        title="Token Usage Costs",
        xaxis_title="Date",
        yaxis=dict(
            title="Daily Cost ($)",
            titlefont=dict(color="rgb(55, 83, 109)"),
            tickfont=dict(color="rgb(55, 83, 109)")
        ),
        yaxis2=dict(
            title="Cumulative Cost ($)",
            titlefont=dict(color="rgb(26, 118, 255)"),
            tickfont=dict(color="rgb(26, 118, 255)"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Convert to JSON for embedding in template
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
from flask import Flask, render_template
from src.app.dashboard.ml_settings import ml_settings_bp
import logging

logger = logging.getLogger(__name__)

def create_flask_app():
    """Create Flask app for admin dashboard"""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = 'your-secret-key'  # Should use environment variables in production
    
    # Register blueprints
    app.register_blueprint(ml_settings_bp)
    
    # Main dashboard route
    @app.route('/')
    def index():
        return render_template('dashboard/index.html', title="Prompt Optimization Dashboard")
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('dashboard/error.html', error=e, title="Not Found"), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('dashboard/error.html', error=e, title="Server Error"), 500
    
    return app
