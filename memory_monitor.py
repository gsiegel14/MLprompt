#!/usr/bin/env python3
"""
Memory Monitor Tool

This script monitors memory usage of the current process and logs it to a file.
It can be used to identify memory leaks and memory-intensive operations.
"""

import os
import psutil
import time
import threading
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_usage.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor memory usage of the current process."""
    
    def __init__(self, interval=5.0, log_file="memory_usage.log"):
        """
        Initialize the memory monitor.
        
        Args:
            interval (float): Interval between checks in seconds
            log_file (str): File to log memory usage to
        """
        self.interval = interval
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.thread = None
        self.start_time = None
        
    def start(self):
        """Start monitoring memory usage."""
        if self.running:
            logger.warning("Memory monitor is already running")
            return
            
        self.running = True
        self.start_time = time.time()
        
        logger.info(f"Starting memory monitoring (interval: {self.interval}s, log: {self.log_file})")
        
        # Log header
        self._log_memory("START")
        
        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop monitoring memory usage."""
        if not self.running:
            logger.warning("Memory monitor is not running")
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        # Log footer
        self._log_memory("STOP")
        
        duration = time.time() - self.start_time
        logger.info(f"Memory monitoring stopped (duration: {duration:.2f}s)")
        
    def _monitor(self):
        """Monitor memory usage in the background."""
        while self.running:
            self._log_memory("MONITOR")
            time.sleep(self.interval)
            
    def _log_memory(self, status):
        """Log current memory usage."""
        try:
            # Get memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # Convert to MB for readability
            rss_mb = memory_info.rss / (1024 * 1024)
            vms_mb = memory_info.vms / (1024 * 1024)
            
            # Log memory usage
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"MEMORY ({status}): RSS: {rss_mb:.2f} MB, VMS: {vms_mb:.2f} MB, Percent: {memory_percent:.2f}%")
            
            # Also write to log file directly
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{status},{rss_mb:.2f},{vms_mb:.2f},{memory_percent:.2f}\n")
                
        except Exception as e:
            logger.error(f"Error logging memory usage: {e}")

# Helper function to get current memory usage
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

# Singleton instance
_monitor = None

def start_monitoring(interval=5.0):
    """Start monitoring memory usage."""
    global _monitor
    if _monitor is None:
        _monitor = MemoryMonitor(interval=interval)
    _monitor.start()
    return _monitor

def stop_monitoring():
    """Stop monitoring memory usage."""
    global _monitor
    if _monitor is not None:
        _monitor.stop()

def log_memory_usage(label):
    """Log current memory usage with a label."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / (1024 * 1024)
    vms_mb = memory_info.vms / (1024 * 1024)
    memory_percent = process.memory_percent()
    logger.info(f"MEMORY ({label}): RSS: {rss_mb:.2f} MB, VMS: {vms_mb:.2f} MB, Percent: {memory_percent:.2f}%")
    return rss_mb

if __name__ == "__main__":
    # Simple demonstration
    logger.info("Starting memory monitoring demo")
    
    # Start monitoring
    monitor = start_monitoring(interval=1.0)
    
    # Allocate some memory
    logger.info("Allocating memory...")
    data = []
    for i in range(10):
        # Add 10MB of data
        data.append(' ' * (10 * 1024 * 1024))
        logger.info(f"Allocated chunk {i+1}/10")
        time.sleep(1)
        
    # Free memory
    logger.info("Freeing memory...")
    data = None
    
    # Wait for garbage collection
    logger.info("Waiting for garbage collection...")
    import gc
    gc.collect()
    time.sleep(2)
    
    # Stop monitoring
    stop_monitoring()
    
    logger.info("Memory monitoring demo complete")