"""
UI layout and styling functions
"""

import streamlit as st
from config.settings import APP_CONFIG

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title=APP_CONFIG["page_title"],
        page_icon=APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"]
    )

def setup_custom_css():
    """Setup custom CSS styling"""
    st.markdown("""
    <style>
    .main-container {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .result-container {
        padding: 2rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .rag-explanation {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        margin-top: 1rem;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .isic-code {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .isic-description {
        font-size: 1.4rem;
        font-style: italic;
        color: #333;
        margin-top: 0.5rem;
    }
    .confidence {
        font-size: 1.4rem;
        color: #FF6B6B;
        font-weight: bold;
    }
    .alt-container {
        padding: 1.5rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
        transition: transform 0.3s ease;
    }
    .alt-container:hover {
        transform: translateY(-2px);
    }
    .mode-selector {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .history-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4CAF50;
        backdrop-filter: blur(5px);
    }
    .stSelectbox > div > div > select {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create sidebar with mode selection and benefits panel"""
    mode = st.sidebar.selectbox("Select Mode", ["Single Description", "Batch Processing"])
    
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px; border-radius: 12px; color: white; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
        <h3 style="text-align: center;">🌟 Key Benefits</h3>
        <ul style="list-style: none; padding-left: 0; line-height: 1.8;">
            <li>✅ <strong>Faster & consistent ISIC coding</strong></li>
            <li>✅ <strong>Reduces human error & workload</strong></li>
            <li>✅ <strong>Supports decision-making</strong> with confidence scores & explanations</li>
            <li>✅ <strong>Batch mode</strong> for large datasets (surveys, censuses)</li>
            <li>✅ <strong>Multilingual support</strong> for international collaboration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    return mode

def display_title_and_description():
    """Display main title and description"""
    st.title("ISIC Classification Tool with RAG")
    st.markdown("""
        This enhanced application uses a BERT model to classify industry (ISIC) descriptions 
        and provides contextual explanations using RAG (Retrieval-Augmented Generation) with Ollama.
        Enter a business activity description, and get both classification and detailed explanations.
    """)

def display_history_panel():
    """Display recent classifications history panel"""
    st.subheader("📈 Recent Classifications")
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    for item in reversed(st.session_state.history[-5:]):
        with st.container():
            st.markdown(f"""
                <div class="history-item">
                    <div style="font-size: 0.8rem; color: #ddd;">"{item['text'][:50]}{'...' if len(item['text']) > 50 else ''}"</div>
                    <div style="font-weight: bold; color: #4CAF50;">{item['code']}</div>
                    <div style="font-size: 0.7rem; color: #FF6B6B;">Confidence: {item['confidence']:.2%}</div>
                    <div style="font-size: 0.7rem; font-style: italic; color: #ccc;">{item['description'][:70]}{'...' if len(item['description']) > 70 else ''}</div>
                </div>
            """, unsafe_allow_html=True)

def display_classification_results(isic_code, display_description, confidence, rag_explanation, rag_system, alternatives):
    """Display classification results with styling"""
    # Display the top result
    st.markdown("### 🎯 Classification Result")
    st.markdown(f"""
        <div class="result-container">
            <p style="color: white; font-size: 1.1rem;">The business description has been classified as:</p>
            <p class="isic-code">{isic_code}</p>
            <p class="isic-description" style="color: white;">{display_description}</p>
            <p class="confidence">Confidence Score: {confidence:.2%}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display RAG explanation
    if rag_explanation and rag_system.ollama_available:
        st.markdown("### 🤖 AI-Powered Explanation")
        st.markdown(f"""
            <div class="rag-explanation">
                <h4 style="color: white;">📖 Context from ISIC Manual</h4>
                <p style="color: white; line-height: 1.6;">{rag_explanation}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display alternative predictions
    st.markdown("### 🔄 Alternative Classifications")
    st.info("The model provides these alternative classifications, ranked by confidence.")
    
    alt_cols = st.columns(2)
    displayed_alternatives = 0
    
    for i, alt in enumerate(alternatives):
        if i == 0 and alt['code'] == isic_code:
            continue
            
        if displayed_alternatives >= 4:
            break
            
        with alt_cols[displayed_alternatives % 2]:
            st.markdown(f"""
                <div class="alt-container">
                    <div style="font-weight: bold; font-size: 1.2rem; color: #333;">{alt['code']}</div>
                    <div style="color: #FF6B6B; font-weight: bold;">Confidence: {alt['confidence']:.2%}</div>
                    <div style="color: #666; font-style: italic; margin-top: 0.5rem;">{alt['description']}</div>
                </div>
            """, unsafe_allow_html=True)
            
        displayed_alternatives += 1

def display_technical_details(isic_code, display_description, confidence, original_lang, description, rag_system):
    """Display technical details in an expander"""
    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
            * **ISIC Code**: {isic_code}
            * **Description**: {display_description}
            * **Confidence**: {confidence:.2%}
            * **Original Language**: {original_lang.upper()}
            * **Business Activity Analyzed**: "{description}"
            * **RAG System**: {'✅ Active' if rag_system.ollama_available else '❌ Ollama not available'}
            * **Document Chunks**: {rag_system.collection.count() if rag_system.collection else 0}
            
            The BERT model analyzed the text description and determined the most likely 
            ISIC classifications. The RAG system provides additional context from the 
            official ISIC manual to explain the classification rationale.
        """)