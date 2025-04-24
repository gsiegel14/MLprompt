from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from app import db
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    # ensure password hash field has length of at least 256
    password_hash = db.Column(db.String(256))
    # Optional fields
    profile_pic = db.Column(db.String(200))
    
    # Relationships
    prompts = db.relationship('Prompt', backref='author', lazy='dynamic')
    optimizations = db.relationship('Optimization', backref='user', lazy='dynamic')
    
    @classmethod
    def get_by_id(cls, user_id):
        """
        Get user by ID with error handling for database connectivity issues
        """
        try:
            return cls.query.get(int(user_id))
        except OperationalError as e:
            logger.error(f"Database connection error in get_by_id: {str(e)}")
            return None
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error in get_by_id: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_by_id: {str(e)}")
            return None
            
    @classmethod
    def get_by_email(cls, email):
        """
        Get user by email with error handling for database connectivity issues
        """
        try:
            return cls.query.filter_by(email=email).first()
        except OperationalError as e:
            logger.error(f"Database connection error in get_by_email: {str(e)}")
            return None
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error in get_by_email: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_by_email: {str(e)}")
            return None
    
    def __repr__(self):
        return f'<User {self.username}>'


class Prompt(db.Model):
    """
    Model for prompts that can be optimized
    """
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    title = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_base = db.Column(db.Boolean, default=False)
    
    # Relationship with optimizations
    optimizations = db.relationship('Optimization', backref='prompt', lazy='dynamic')
    
    def __repr__(self):
        return f'<Prompt {self.title}>'


class Optimization(db.Model):
    """
    Model for storing optimized prompts
    """
    id = db.Column(db.Integer, primary_key=True)
    original_prompt_id = db.Column(db.Integer, db.ForeignKey('prompt.id'), nullable=False)
    optimized_content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    base_prompts_used = db.Column(db.Text, nullable=True)  # Stores base prompts used for optimization
    
    def __repr__(self):
        return f'<Optimization {self.id} for Prompt {self.original_prompt_id}>'