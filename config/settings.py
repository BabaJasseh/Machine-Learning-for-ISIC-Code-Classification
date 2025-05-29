"""
Configuration settings for the ISIC Classification Tool
"""

# Application configuration
APP_CONFIG = {
    "page_title": "ISIC Classification Tool with RAG",
    "page_icon": "📊",
    "layout": "wide"
}

# RAG Configuration
RAG_CONFIG = {
    "ollama_base_url": "http://localhost:11434",  # Default Ollama URL
    "model_name": "llama3.2",  # 
    "embedding_model": "all-MiniLM-L6-v2",
    "chroma_persist_path": "chroma_isic_storage",
    "pdf_path": "isic_manual/ISIC_SUMMARY_MANUAL.pdf"
}

# model types
MODEL_CONFIG = {
    "single": {
        "model_type": "bert",
        "model_path": "experiments/celestial-sweep-1-lszxepju/best_model"  # replace 'celestial-sweep-1-lszxepju' with <your-sweep-id>
    },
    "batch": {
        "model_type": "bert",
        "model_path": "experiments/celestial-sweep-1-lszxepju/best_model"  # replace 'celestial-sweep-1-lszxepju' with <your-sweep-id>
    }
}

# File paths
FILE_PATHS = {
    "classes": "./data/classes.npy",
    "isic_csv": "./data/isic_gam.csv"  # replace with yours
}

# lang config
LANGUAGE_NAMES = {
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "th": "Thai",
    "vi": "Vietnamese", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian",
    "sk": "Slovak", "sl": "Slovenian", "et": "Estonian", "lv": "Latvian",
    "lt": "Lithuanian", "mt": "Maltese", "ga": "Irish", "cy": "Welsh"
}

# Supported encodings for CSV files
ENCODINGS = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']