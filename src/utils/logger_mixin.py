"""
Logger mixin for logging to a file and console from any class.
"""

import logging
import os

class LoggerMixin:

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            # Get logger named after the class
            self._logger = logging.getLogger(self.__class__.__name__)
            
            if not self._logger.handlers:  # Avoid adding handlers multiple times
                # Setup logging
                self._logger.setLevel(logging.INFO)
                
                # Ensure log directory exists
                os.makedirs("data/logs", exist_ok=True)
                
                # Create file handler
                file_handler = logging.FileHandler("data/logs/application.log")
                file_handler.setLevel(logging.INFO)
                
                # Create formatter
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                
                # Add handler
                self._logger.addHandler(file_handler)
                
        return self._logger 