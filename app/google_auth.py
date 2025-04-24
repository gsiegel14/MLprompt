import json
import os

import requests
from flask import Blueprint, redirect, request, url_for, flash, current_app, render_template
from flask_login import login_required, login_user, logout_user, current_user
from app.models import User
from app import db, logger
from oauthlib.oauth2 import WebApplicationClient

# Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# OAuth client setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Blueprint
google_auth = Blueprint("google_auth", __name__)

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

@google_auth.route("/")
def index():
    """
    Home page route. If user is not logged in, redirect to login.
    """
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('login'))

@google_auth.route("/login")
def login():
    """
    Google login route. Redirects to Google's OAuth consent screen.
    """
    # Find out what URL to hit for Google login
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Get the absolute URL for the callback
    callback_url = url_for("google_auth.callback", _external=True)
    
    # In development, we need to use the actual domain for OAuth callbacks
    if os.environ.get("REPLIT_DEV_DOMAIN"):
        callback_url = f"https://{os.environ.get('REPLIT_DEV_DOMAIN')}/google_auth/callback"
        # Print the exact callback URL for debugging
        print(f"Using callback URL: {callback_url}")
    elif os.environ.get("REPLIT_DOMAIN"):
        callback_url = f"https://{os.environ.get('REPLIT_DOMAIN')}/google_auth/callback"
        print(f"Using callback URL: {callback_url}")
    else:
        # For local development
        callback_url = callback_url.replace("http://", "https://")
    
    # Use library to construct the request for Google login
    # Request offline access (refresh_token) and user's profile and email
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=callback_url,
        scope=["openid", "email", "profile"],
    )
    
    # Redirect to Google's OAuth consent screen
    return redirect(request_uri)

@google_auth.route("/callback")
def callback():
    """
    Google OAuth callback route. Handles the response from Google.
    """
    # Get authorization code Google sent back
    code = request.args.get("code")
    
    # If there's no code, the user may have accessed this URL directly
    # or there may have been an error in the OAuth process
    if code is None:
        error = request.args.get("error", "Unknown error")
        logger.error(f"OAuth error: {error}")
        flash(f"Authentication failed: {error}. Please try again.", "danger")
        return redirect(url_for("login"))
    
    # Find out what URL to hit to get tokens
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]
    
    # Prepare and send a request to get tokens
    # Get the absolute URL for the callback again
    callback_url = url_for("google_auth.callback", _external=True)
    
    # In development, we need to use the actual domain for OAuth callbacks
    if os.environ.get("REPLIT_DEV_DOMAIN"):
        callback_url = f"https://{os.environ.get('REPLIT_DEV_DOMAIN')}/google_auth/callback"
        # Print the exact callback URL for debugging
        print(f"Callback function using URL: {callback_url}")
    elif os.environ.get("REPLIT_DOMAIN"):
        callback_url = f"https://{os.environ.get('REPLIT_DOMAIN')}/google_auth/callback"
        print(f"Callback function using URL: {callback_url}")
    else:
        # For local development
        callback_url = callback_url.replace("http://", "https://")
    
    # Make sure we use the correct domain in the authorization response
    authorization_response = request.url
    
    # Extract just the path portion and use the correct domain
    path = request.path
    query = request.query_string.decode('utf-8')
    query_part = f"?{query}" if query else ""
    
    if os.environ.get("REPLIT_DEV_DOMAIN"):
        authorization_response = f"https://{os.environ.get('REPLIT_DEV_DOMAIN')}{path}{query_part}"
        print(f"Using authorization_response: {authorization_response}")
    elif os.environ.get("REPLIT_DOMAIN"):
        authorization_response = f"https://{os.environ.get('REPLIT_DOMAIN')}{path}{query_part}"
        print(f"Using authorization_response: {authorization_response}")
    else:
        # For local development
        authorization_response = authorization_response.replace("http://", "https://")
    
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=authorization_response,
        redirect_url=callback_url,
        code=code
    )
    # Use auth only if both ID and SECRET are available as strings
    try:
        if isinstance(GOOGLE_CLIENT_ID, str) and isinstance(GOOGLE_CLIENT_SECRET, str):
            auth = (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
            token_response = requests.post(
                token_url,
                headers=headers,
                data=body,
                auth=auth,
            )
        else:
            # Log the issue with credentials
            logger.warning("Google OAuth credentials are missing or invalid. Attempting without auth.")
            token_response = requests.post(
                token_url,
                headers=headers,
                data=body,
            )
        
        # Log the response for debugging
        logger.debug(f"Token response status code: {token_response.status_code}")
        
        # Check if the response is valid
        if token_response.status_code != 200:
            logger.error(f"Token response error: {token_response.text}")
            flash("Authentication failed. Please try again.", "danger")
            return redirect(url_for("login"))
        
        # Parse the tokens
        client.parse_request_body_response(json.dumps(token_response.json()))
        
    except Exception as e:
        logger.error(f"Error in OAuth flow: {str(e)}")
        flash("Authentication error. Please try again.", "danger")
        return redirect(url_for("login"))
    
    # Get user info from Google
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)
    
    # Verify user's email is verified by Google
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json().get("picture", None)
        users_name = userinfo_response.json().get("given_name", users_email.split('@')[0])
    else:
        flash("User email not verified by Google.", "danger")
        return redirect(url_for("google_auth.login"))
    
    # Create or update user
    try:
        # Use our new error-handling method
        user = User.get_by_email(users_email)
        if not user:
            # Check if there's already a user with the same username
            # Wrap in try-except for database resilience
            try:
                existing_user_with_username = User.query.filter_by(username=users_name).first()
                if existing_user_with_username:
                    # Append a unique identifier to avoid username conflicts
                    users_name = f"{users_name}_{unique_id[:6]}"
            except Exception as username_check_error:
                logger.warning(f"Error checking for username conflicts: {str(username_check_error)}")
                # If we can't check, make it unique anyway as a precaution
                users_name = f"{users_name}_{unique_id[:6]}"
            
            # Create a new user with retries
            max_retries = 3
            retry_count = 0
            user_created = False
            
            while retry_count < max_retries and not user_created:
                try:
                    # Create a new user
                    user = User(
                        username=users_name,
                        email=users_email,
                        profile_pic=picture
                    )
                    db.session.add(user)
                    db.session.commit()
                    logger.info(f"Created new user: {users_email}")
                    user_created = True
                except Exception as retry_error:
                    retry_count += 1
                    logger.warning(f"Database error during user creation (attempt {retry_count}): {str(retry_error)}")
                    db.session.rollback()
                    if retry_count < max_retries:
                        # Wait briefly before retrying
                        import time
                        time.sleep(0.5)
            
            if not user_created:
                logger.error("Failed to create user after maximum retries")
                flash("Database connection issue. Please try again in a moment.", "danger")
                return redirect(url_for("login"))
        else:
            # Update existing user with retries
            max_retries = 3
            retry_count = 0
            user_updated = False
            
            while retry_count < max_retries and not user_updated:
                try:
                    # Update existing user
                    user.username = users_name
                    user.profile_pic = picture
                    db.session.commit()
                    logger.info(f"Updated existing user: {users_email}")
                    user_updated = True
                except Exception as retry_error:
                    retry_count += 1
                    logger.warning(f"Database error during user update (attempt {retry_count}): {str(retry_error)}")
                    db.session.rollback()
                    if retry_count < max_retries:
                        # Wait briefly before retrying
                        import time
                        time.sleep(0.5)
            
            if not user_updated:
                logger.error("Failed to update user after maximum retries")
                # If we can't update, we can still try to login with existing data
                flash("Warning: User profile could not be updated, but login will proceed.", "warning")
        
        # Log in the user
        login_user(user)
        logger.info(f"Logged in user: {users_email}")
    
    except Exception as e:
        logger.error(f"Database error during user login process: {str(e)}")
        db.session.rollback()  # Roll back the transaction on error
        flash("An error occurred during login. Please try again.", "danger")
        return redirect(url_for("login"))
    
    # Redirect to main dashboard
    return redirect(url_for("dashboard"))

@google_auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))