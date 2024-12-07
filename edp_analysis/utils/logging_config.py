"""
Logging configuration for the NFL EDP analysis codebase.
"""

import logging

def setup_logging(log_level, module_name=None):
    """
    Set up logging for a module.
    
    Args:
        log_level (int): The logging level to use.
        module_name (str, optional): The name of the module. Defaults to None.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    logger = setup_logging(logging.DEBUG, module_name="test_module")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message") 