from flask import render_template, redirect, url_for, flash, request, session, Blueprint
import logging
from app import db
from app.models import Prompt, Optimization
from prompts import optimizer_output_prompt, optimizer_enhanced_prompt
from app.utils import substitute_variables

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a Blueprint for main routes
main_blueprint = Blueprint('main', __name__)

def index():
    """Render the home page"""
    return render_template('index.html')

def prompts():
    """View and create prompts"""
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        is_base = 'is_base' in request.form
        
        # Simple validation
        if not title or not content:
            flash('Title and content are required', 'error')
            return redirect(url_for('prompts'))
        
        # Create new prompt
        new_prompt = Prompt(
            title=title,
            content=content,
            is_base=is_base,
            user_id=1  # For demo purposes
        )
        db.session.add(new_prompt)
        db.session.commit()
        
        flash('Prompt created successfully', 'success')
        return redirect(url_for('prompts'))
    
    # Get all prompts, with base prompts first
    all_prompts = Prompt.query.order_by(Prompt.is_base.desc(), Prompt.created_at.desc()).all()
    return render_template('prompts.html', prompts=all_prompts)

def optimize():
    """Optimize a prompt with base prompts"""
    if request.method == 'POST':
        prompt_id = request.form.get('prompt_id')
        selected_base_prompts = request.form.getlist('base_prompts')
        
        # Get the original prompt
        original_prompt = Prompt.query.get(prompt_id)
        if not original_prompt:
            flash('Prompt not found', 'error')
            return redirect(url_for('optimize'))
        
        # Get all selected base prompts
        base_prompts = Prompt.query.filter(Prompt.id.in_(selected_base_prompts)).all()
        base_prompts_text = "\n\n".join([prompt.content for prompt in base_prompts])
        
        # API CALL 1: Run user input with base prompts
        # In a real implementation, this would call an actual API
        logging.debug("API CALL 1: Running user input with base prompts")
        # Simulate API call 1 output for demonstration
        base_response = f"Base response for: {original_prompt.content[:50]}..."
        
        # API CALL 2: Hugging Face evaluation of base prompt results
        logging.debug("API CALL 2: Hugging Face evaluation of base prompt results")
        # Simulate HF evaluation for demonstration
        eval_data_base = """
        Evaluation of base prompt:
        - Accuracy: 7/10
        - Completeness: 6/10
        - Relevance: 8/10
        
        Issues identified:
        - Missing specific constraints
        - Could be more detailed in guidance
        - Structure could be improved
        """
        
        # Setup variables for the optimization
        variables = {
            'BASE_PROMPTS': base_prompts_text,
            'EVAL_DATA_BASE': eval_data_base
        }
        
        # API CALL 3: Optimize prompts based on evaluation data
        logging.debug("API CALL 3: Optimize prompts based on evaluation data")
        # Use the enhanced optimizer prompt that includes evaluation data
        full_optimization_prompt = substitute_variables(optimizer_enhanced_prompt, variables)
        logging.debug(f"Optimization prompt with substituted variables: {full_optimization_prompt}")
        
        # In a real implementation, this would call an API to optimize the prompt
        # For demo purposes, create a simulated optimized result with system message and output prompt
        optimized_system_message = f"OPTIMIZED SYSTEM MESSAGE:\n\nYou are an expert AI assistant focused on {original_prompt.title}. Follow these guidelines carefully."
        optimized_output_prompt = f"OPTIMIZED OUTPUT PROMPT:\n\n{original_prompt.content}\n\nProvide detailed, accurate information with proper citations."
        
        # Combined optimized prompts
        optimized_content = f"===SYSTEM_MESSAGE===\n{optimized_system_message}\n\n===OUTPUT_PROMPT===\n{optimized_output_prompt}"
        
        # API CALL 4: Run user input with optimized prompts
        logging.debug("API CALL 4: Run user input with optimized prompts")
        # Simulate API call 4 output for demonstration
        optimized_response = f"Optimized response for: {original_prompt.content[:50]}..."
        
        # API CALL 5: Hugging Face evaluation of optimized prompt results
        logging.debug("API CALL 5: Hugging Face evaluation of optimized prompt results")
        # Simulate HF evaluation for demonstration
        eval_data_optimized = """
        Evaluation of optimized prompt:
        - Accuracy: 9/10
        - Completeness: 8/10
        - Relevance: 9/10
        
        Improvements:
        - Better structured response
        - More comprehensive coverage
        - Clearer guidance for the model
        """
        
        # Store the optimization
        new_optimization = Optimization(
            original_prompt_id=prompt_id,
            optimized_content=optimized_content,
            user_id=1,  # For demo purposes
            base_prompts_used=base_prompts_text
        )
        db.session.add(new_optimization)
        db.session.commit()
        
        return render_template('optimize_result.html', 
                              original=original_prompt, 
                              optimized=optimized_content,
                              optimized_system_message=optimized_system_message,
                              optimized_output_prompt=optimized_output_prompt,
                              base_prompts=base_prompts,
                              eval_data_base=eval_data_base,
                              eval_data_optimized=eval_data_optimized)
    
    # GET request
    prompts = Prompt.query.filter_by(is_base=False).all()
    base_prompts = Prompt.query.filter_by(is_base=True).all()
    
    return render_template('optimize.html', prompts=prompts, base_prompts=base_prompts)

def view_optimizations():
    """View all optimizations"""
    optimizations = Optimization.query.order_by(Optimization.created_at.desc()).all()
    return render_template('optimizations.html', optimizations=optimizations)

def login():
    """Login page"""
    if request.method == 'POST':
        # For demo purposes, we'll just redirect to the home page
        flash('Login successful!', 'success')
        return redirect(url_for('index'))
    return render_template('login.html')

def variables():
    """View variable documentation"""
    # This page will display documentation about the variables used in the optimization process
    return render_template('variables.html')