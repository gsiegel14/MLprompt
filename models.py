from app import db
from flask_login import UserMixin
from datetime import datetime


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    # ensure password hash field has length of at least 256
    password_hash = db.Column(db.String(256))
    prompts = db.relationship('Prompt', backref='author', lazy='dynamic')
    optimizations = db.relationship('Optimization', backref='user', lazy='dynamic')


class Prompt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    title = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_base = db.Column(db.Boolean, default=False)
    optimizations = db.relationship('Optimization', backref='prompt', lazy='dynamic')


class Optimization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_prompt_id = db.Column(db.Integer, db.ForeignKey('prompt.id'), nullable=False)
    optimized_content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    base_prompts_used = db.Column(db.Text, nullable=True)  # Stores base prompts used for optimization
