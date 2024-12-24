"""
Logging Configuration Module

This module provides a standard logging configuration for the EDP analysis package.
"""

import logging
import sys


def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up logging configuration for the package."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler) 