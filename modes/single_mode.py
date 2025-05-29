
"""
Single description classification mode
"""

import streamlit as st
from models.loader import load_model_and_data
from rag.system import initialize_rag_system
from utils.classification import classify_description
from utils.language import detect_language, translate_text
from utils.speech import speech_to_text
from ui.layout import (
    display_title_and_description, 
    display_history_panel, 
    display_classification_results,
    display_technical_details
)


def single_classification_mode():
    """Main function for single description classification mode"""
    # Display title and description
    display_title_and_description()
    
    # Load models and data at startup
    try:
        with st.spinner("Loading BERT classification model and ISIC data..."):
            model, label_encoder, isic_dict = load_model_and_data("single")
        st.success("✅ Classification model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading classification resources: {e}")
        st.stop()
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        classify_button, description = create_input_section()
        
    with col2:
        display_history_panel()
    
    # Handle classification
    handle_classification_logic(model, label_encoder, isic_dict, rag_system, classify_button, description)


def create_input_section():
    """Create the input section for business descriptions"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.subheader("🔍 Enter Business Description")
    st.write("""
        You can type descriptions in multiple languages. 
        The app automatically detects and translates them, providing responses in your original language.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input method tabs
    input_method = st.radio(
        "Choose input method:",
        ["📝 Text Input", "🎙️ Voice Input"],
        horizontal=True
    )

    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    
    # Handle input method
    if input_method == "📝 Text Input":
        description = st.text_area(
            "Describe the business activity in detail",
            value=st.session_state.text_input,
            height=150,
            placeholder="Example: Manufacturing of wooden furniture for household use"
        )
        st.session_state.text_input = description
    else:
        speech_text = speech_to_text()
        
        if speech_text:
            st.text_area(
                "Transcribed text (edit if needed):",
                value=speech_text,
                key="transcribed_text",
                height=150
            )
            description = st.session_state.transcribed_text
            st.session_state.text_input = description
        else:
            if "transcribed_text" in st.session_state:
                description = st.session_state.transcribed_text
            else:
                description = ""
                
            if description:
                description = st.text_area(
                    "Transcribed text (edit if needed):",
                    value=description,
                    key="edit_transcribed",
                    height=150
                )
    
    return st.button("🚀 Classify", type="primary", use_container_width=True), description


def handle_classification_logic(model, label_encoder, isic_dict, rag_system, classify_button, description):
    """Handle the classification logic and display results"""
    if classify_button and description:
        with st.spinner("🔄 Classifying and generating explanation..."):
            progress_bar = st.progress(0)
            
            # Language detection and translation
            lang = detect_language(description)
            original_lang = lang
            progress_bar.progress(20)
            
            if lang != "en":
                st.info(f"🌍 Detected language: {lang.upper()}. Processing...")
                translated_desc = translate_text(description, source_lang=lang, target_lang="en")
            else:
                translated_desc = description
            
            progress_bar.progress(40)
            
            # Run classification
            isic_code, confidence, description_text, alternatives = classify_description(
                translated_desc, model, label_encoder, isic_dict
            )
            progress_bar.progress(60)
            
            # Translate results back if needed
            display_description = description_text
            if lang != "en":
                display_description = translate_text(description_text, source_lang="en", target_lang=lang)
                for alt in alternatives:
                    alt['description'] = translate_text(alt['description'], source_lang="en", target_lang=lang)
            
            progress_bar.progress(80)
            
            # Generate RAG explanation in original language
            rag_explanation = rag_system.generate_explanation(
                isic_code, description_text, description, original_lang
            )
            progress_bar.progress(100)
            
            # Add to history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'text': description,
                'code': isic_code,
                'confidence': confidence,
                'description': display_description if display_description else "Description not available"
            })
        
        # Display results
        display_classification_results(
            isic_code, display_description, confidence, 
            rag_explanation, rag_system, alternatives
        )
        
        # Show technical details
        display_technical_details(
            isic_code, display_description, confidence, 
            original_lang, description, rag_system
        )
        
    elif classify_button and not description:
        st.warning("⚠️ Please enter a business description to classify.")
