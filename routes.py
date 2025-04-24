from flask import render_template, request, redirect, url_for, flash, jsonify, session
from app import app, db
from models import User, Prompt, Optimization
from utils import substitute_variables
from prompts import optimizer_output_prompt
import logging

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/prompts', methods=['GET', 'POST'])
def prompts():
    """View and create prompts"""
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        is_base = request.form.get('is_base') == 'on'
        
        # For demo purposes, assuming user_id = 1
        user_id = 1
        
        new_prompt = Prompt(title=title, content=content, user_id=user_id, is_base=is_base)
        db.session.add(new_prompt)
        db.session.commit()
        
        flash('Prompt created successfully!', 'success')
        return redirect(url_for('prompts'))
    
    # Get all prompts, with base prompts first
    all_prompts = Prompt.query.order_by(Prompt.is_base.desc(), Prompt.created_at.desc()).all()
    return render_template('prompts.html', prompts=all_prompts)

@app.route('/optimize', methods=['GET', 'POST'])
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
        
        # For demonstration, creating a sample optimized version
        # In a real application, this would call an API to optimize the prompt
        # using the optimizer_output_prompt with substituted variables
        
        variables = {
            'BASE_PROMPTS': base_prompts_text
        }
        
        # Here we'd call an external API with the substituted prompt
        full_optimization_prompt = substitute_variables(optimizer_output_prompt, variables)
        logging.debug(f"Optimization prompt with substituted variables: {full_optimization_prompt}")
        
        # For demo, just append "OPTIMIZED" to the original
        optimized_content = f"OPTIMIZED VERSION:\n\n{original_prompt.content}"
        
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
                              base_prompts=base_prompts)
    
    # GET request
    prompts = Prompt.query.filter_by(is_base=False).all()
    base_prompts = Prompt.query.filter_by(is_base=True).all()
    
    return render_template('optimize.html', prompts=prompts, base_prompts=base_prompts)

@app.route('/optimizations')
def view_optimizations():
    """View all optimizations"""
    optimizations = Optimization.query.order_by(Optimization.created_at.desc()).all()
    return render_template('optimizations.html', optimizations=optimizations)
