from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from app import db
import logging

logger = logging.getLogger(__name__)

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    # Optional fields
    profile_pic = db.Column(db.String(200))
    
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