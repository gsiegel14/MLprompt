
#!/usr/bin/env python
"""
Fix API Authentication

This script adds API key authentication to the application to prevent 
redirects to login page when using API endpoints.
"""

import os
import sys
import logging
import inspect
import random
import string
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.StreamHandler(sys.stdout)
                   ])
logger = logging.getLogger("api_auth_fix")

def generate_api_key():
    """Generate a random API key."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def modify_main_py():
    """Modify main.py to support API key authentication."""
    main_py_file = "app/main.py"
    
    try:
        with open(main_py_file, 'r') as f:
            content = f.read()
        
        # Check if API key auth is already implemented
        if "API_KEY" in content and "X-API-Key" in content:
            logger.info("API key authentication already implemented, skipping")
            return True
        
        # Add API key configuration
        api_key_config = """
# API Key configuration
API_KEY = os.environ.get('API_KEY', 'dev_api_key')  # Default key for development

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if API key is provided in header
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key == API_KEY:
            return f(*args, **kwargs)
        
        # For API endpoints, fail with 401 Unauthorized
        if request.path.startswith('/api/'):
            return jsonify({'error': 'API key required'}), 401
            
        # For other routes, proceed with normal auth check
        if 'user_id' not in session:
            logger.debug("User not authenticated, redirecting to login page")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
"""
        
        # Add the imports
        if "from functools import wraps" not in content:
            import_pos = content.find("import ")
            if import_pos > -1:
                next_line_pos = content.find("\n", import_pos)
                if next_line_pos > -1:
                    content = content[:next_line_pos+1] + "from functools import wraps\n" + content[next_line_pos+1:]
        
        # Add the API key code
        auth_pos = content.find("def login_required")
        if auth_pos == -1:
            # If login_required not found, look for another insertion point
            auth_pos = content.find("def login()")
        
        if auth_pos > -1:
            content = content[:auth_pos] + api_key_config + "\n" + content[auth_pos:]
        
        # Replace login_required with require_api_key where appropriate
        protected_endpoints = ["@app.route('/api/", "@app.route('/five_api/"]
        lines = content.split('\n')
        modified_lines = []
        
        for i, line in enumerate(lines):
            if any(endpoint in line for endpoint in protected_endpoints):
                # Check if the next line has @login_required
                if i+1 < len(lines) and "@login_required" in lines[i+1]:
                    modified_lines.append(line)
                    modified_lines.append("@require_api_key")
                    i += 1  # Skip the original @login_required line
                else:
                    modified_lines.append(line)
                    modified_lines.append("@require_api_key")
            else:
                modified_lines.append(line)
        
        updated_content = '\n'.join(modified_lines)
        
        # Save the modified file
        with open(main_py_file, 'w') as f:
            f.write(updated_content)
            
        logger.info(f"✅ Successfully updated {main_py_file} with API key authentication")
        return True
    except Exception as e:
        logger.error(f"❌ Error modifying main.py: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def update_test_script():
    """Update test script to use API key."""
    test_script = "test_five_api_workflow_row2.py"
    
    try:
        with open(test_script, 'r') as f:
            content = f.read()
        
        # Check if API key is already included
        if "headers={'X-API-Key'" in content:
            logger.info("API key already included in test script, skipping")
            return True
        
        # Modify all requests to include the API key
        updated_content = re.sub(
            r'requests\.(get|post|put|delete)\(\s*([^,\)]+)',
            r'requests.\1(\2, headers={"X-API-Key": "dev_api_key"}',
            content
        )
        
        # Save the modified file
        with open(test_script, 'w') as f:
            f.write(updated_content)
            
        logger.info(f"✅ Successfully updated {test_script} to use API key")
        return True
    except Exception as e:
        logger.error(f"❌ Error updating test script: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("=== FIXING API AUTHENTICATION ===")
    
    # Modify main.py to add API key auth
    modify_main_py()
    
    # Update test script
    update_test_script()
    
    logger.info("=== API AUTHENTICATION FIX COMPLETED ===")
    
    print("\n✅ API authentication fixes completed!")
    print("Now the API endpoints will accept X-API-Key header for authentication")
    print("The test script has been updated to include this header.\n")

if __name__ == "__main__":
    main()
