"""
Language processing utilities for detection and translation
"""

import streamlit as st
from langdetect import detect
from deep_translator import GoogleTranslator

def detect_language(text):
    """Detect the language of input text"""
    try:
        return detect(text)
    except:
        return "unknown"

def translate_text(text, source_lang="auto", target_lang="en"):
    """Translate text from source language to target language"""
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text