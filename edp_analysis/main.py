"""
Main script to run the NFL EDP analysis pipeline.
"""

from data_pipeline import NFLDataPipeline
from utils.logging_config import setup_logging
import logging

logger = setup_logging(logging.INFO, module_name=__name__)

def main():
    try:
        # Initialize pipeline
        pipeline = NFLDataPipeline(
            seasons=[2023, 2024],
            log_level=logging.INFO
        )
        
        # Run pipeline
        df = pipeline.run_pipeline()
        logger.info(f"Pipeline completed successfully. Final dataset shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return None

if __name__ == "__main__":
    df = main()