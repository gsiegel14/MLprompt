
import os
import shutil
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
MAX_CACHE_SIZE_MB = 100  # Maximum cache size in MB
MAX_CACHE_AGE_DAYS = 7   # Maximum age of cache files in days

def setup_cache_dir(cache_dir_path="src/cache", max_size_mb=MAX_CACHE_SIZE_MB, max_age_days=MAX_CACHE_AGE_DAYS):
    """Setup and manage cache directory with size and age limits"""
    # Create cache directory if it doesn't exist
    cache_dir = Path(cache_dir_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # For deployment, use more aggressive cache management
    if os.environ.get('DEPLOYMENT_MODE') == 'production':
        max_size_mb = 50  # Only 50MB in production
        max_age_days = 1   # Only 1 day in production
    
    # Check current cache size
    try:
        cache_size_mb = sum(f.stat().st_size for f in cache_dir.glob('**/*') if f.is_file()) / (1024 * 1024)
        
        # If cache is too large, clean it
        if cache_size_mb > max_size_mb:
            logger.info(f"Cache size ({cache_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB). Cleaning cache.")
            clean_cache_by_size(cache_dir, max_size_mb)
        
        # Clean old files regardless of size
        clean_cache_by_age(cache_dir, max_age_days)
        
        # Log final cache size
        current_size_mb = sum(f.stat().st_size for f in cache_dir.glob('**/*') if f.is_file()) / (1024 * 1024)
        logger.info(f"Current cache size: {current_size_mb:.2f}MB")
        
    except Exception as e:
        logger.error(f"Error managing cache: {e}")
    
    return cache_dir

def clean_cache_by_size(cache_dir, max_size_mb):
    """Clean cache by removing oldest files until we're under the size limit"""
    try:
        # Get all files with their last access time
        files = []
        for file_path in cache_dir.glob('**/*'):
            if file_path.is_file():
                files.append((file_path, file_path.stat().st_atime))
        
        # Sort by access time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Delete oldest files until we're under the limit
        current_size_mb = sum(f[0].stat().st_size for f in files) / (1024 * 1024)
        deleted_count = 0
        
        for file_path, _ in files:
            if current_size_mb <= max_size_mb * 0.8:  # Target 80% of limit
                break
                
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            try:
                file_path.unlink()
                current_size_mb -= file_size_mb
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        
        logger.info(f"Removed {deleted_count} files from cache")
    except Exception as e:
        logger.error(f"Error cleaning cache by size: {e}")

def clean_cache_by_age(cache_dir, max_age_days):
    """Clean cache by removing files older than max_age_days"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        deleted_count = 0
        
        for file_path in cache_dir.glob('**/*'):
            if file_path.is_file():
                file_age_seconds = current_time - file_path.stat().st_mtime
                if file_age_seconds > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting old file {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Removed {deleted_count} old files from cache")
    except Exception as e:
        logger.error(f"Error cleaning cache by age: {e}")

# Usage in application startup
def initialize_cache():
    """Initialize the cache system"""
    logger.info("Initializing cache system...")
    cache_dir = setup_cache_dir()
    logger.info(f"Cache directory {cache_dir} initialized")
    return cache_dir

if __name__ == "__main__":
    initialize_cache()
