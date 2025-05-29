"""
ISIC Classification Tool with RAG - Main Application
Entry point for the Streamlit application
"""

import streamlit as st
import os
from config.settings import APP_CONFIG, RAG_CONFIG
from ui.layout import setup_page_config, setup_custom_css, create_sidebar
from modes.single_mode import single_classification_mode
from modes.batch_mode import batch_processing_mode

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """Main application entry point"""
    # Setup page configuration
    setup_page_config()
    
    # Setup custom CSS
    setup_custom_css()
    
    # Create sidebar and get selected mode
    mode = create_sidebar()
    
    # Route to appropriate mode
    if mode == "Single Description":
        single_classification_mode()
    else:
        batch_processing_mode()

if __name__ == "__main__":
    main()