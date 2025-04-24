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
        return redirect(url_for('index'))
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
    
    # Ensure HTTPS for the callback URL
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
    
    # Find out what URL to hit to get tokens
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]
    
    # Prepare and send a request to get tokens
    # Get the absolute URL for the callback again
    callback_url = url_for("google_auth.callback", _external=True)
    callback_url = callback_url.replace("http://", "https://")
    
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url.replace("http://", "https://"),
        redirect_url=callback_url,
        code=code
    )
    # Use auth only if both ID and SECRET are available as strings
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

    # Parse the tokens
    client.parse_request_body_response(json.dumps(token_response.json()))
    
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
    user = User.query.filter_by(email=users_email).first()
    if not user:
        # Create a new user
        user = User(
            username=users_name,
            email=users_email,
            profile_pic=picture
        )
        db.session.add(user)
        db.session.commit()
        logger.info(f"Created new user: {users_email}")
    else:
        # Update existing user
        user.username = users_name
        user.profile_pic = picture
        db.session.commit()
        logger.info(f"Updated existing user: {users_email}")
    
    # Log in the user
    login_user(user)
    logger.info(f"Logged in user: {users_email}")
    
    # Redirect to main dashboard
    return redirect(url_for("dashboard"))

@google_auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))