import pandas as pd
import numpy as np
import streamlit as st
import os
import time
import torch
from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax
import speech_recognition as sr
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="ISIC Classification Tool",
    page_icon="📊",
    layout="wide"
)

# Application title and description
st.title("ISIC Classification Tool")
st.markdown("""
    This application uses a BERT model to classify industry (ISIC) descriptions. 
    Enter a business activity description (or use voice input), and the model will predict 
    the appropriate ISIC classification code.
""")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define model types for use throughout the app - now both single and batch use the same model
MODEL_CONFIG = {
    "single": {
        "model_type": "bert",
        "model_path": "experiments/good-sweep-49wtfpcwu/best_model"
    },
    "batch": {
        "model_type": "bert",
        "model_path": "experiments/good-sweep-49wtfpcwu/best_model"
    }
}

# Function to load models and ISIC data
@st.cache_resource
def load_model_and_data(mode="single"):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    cuda_device = 0 if use_cuda else -1
    
    # Load the label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    
    # Select model configuration based on mode
    model_config = MODEL_CONFIG[mode]
    
    # Load the model
    model = ClassificationModel(
        model_config["model_type"],
        model_config["model_path"],
        use_cuda=use_cuda,
        cuda_device=cuda_device
    )
    
    # Load ISIC codes and descriptions
    try:
        # Try multiple encodings to handle potential encoding issues
        encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                isic_df = pd.read_csv('isic_perfect.csv', encoding=encoding)
                print(f"Successfully loaded ISIC data with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            # If all encodings fail, raise an exception
            raise Exception("Could not read CSV file with any of the attempted encodings")
        
        # Clean and standardize the CLASS column
        isic_df['CLASS'] = isic_df['CLASS'].astype(str).str.strip()
        
        # Create a dictionary for quick lookup
        isic_dict = dict(zip(isic_df['CLASS'], isic_df['ECONOMICS ACTIVITIES']))
        
        # Print debug info about dictionary size
        print(f"Loaded {len(isic_dict)} ISIC codes and descriptions")
        print(f"Sample codes: {list(isic_dict.keys())[:5]}")
        
    except Exception as e:
        st.error(f"Error loading ISIC data: {e}")
        isic_dict = {}
    
    return model, label_encoder, isic_dict

# Function to classify a single description with alternatives
def classify_description(text, model, label_encoder, isic_dict):
    # Make prediction
    prediction, raw_output = model.predict([text])
    
    # Get softmax scores for all classes
    scores = softmax(raw_output[0])
    
    # Get top prediction
    top_pred_idx = prediction[0]
    top_confidence = scores[top_pred_idx]
    top_isic_code = label_encoder.inverse_transform([top_pred_idx])[0]
    
    # Clean and standardize the ISIC code for lookup
    top_isic_code_clean = str(top_isic_code).strip()
    
    # Debug information about the code lookup
    print(f"Looking up ISIC code: '{top_isic_code_clean}'")
    print(f"Available keys (sample): {list(isic_dict.keys())[:5]}")
    
    # Get the description for the top prediction
    # Try with different formattings if needed
    if top_isic_code_clean in isic_dict:
        top_description = isic_dict.get(top_isic_code_clean)
    else:
        # Try padding with zeros (e.g., '42' -> '0042')
        padded_code = top_isic_code_clean.zfill(4)
        if padded_code in isic_dict:
            top_description = isic_dict.get(padded_code)
        else:
            # As a fallback, try to find a close match
            for key in isic_dict.keys():
                if top_isic_code_clean in key or key in top_isic_code_clean:
                    top_description = isic_dict.get(key)
                    break
            else:
                top_description = "Description not found"
    
    # Get top 5 alternatives (including the top prediction)
    top_indices = np.argsort(scores)[::-1][:5]
    alternatives = []
    
    for idx in top_indices:
        isic_code = label_encoder.inverse_transform([idx])[0]
        confidence = scores[idx]
        
        # Clean and standardize for lookup
        isic_code_clean = str(isic_code).strip()
        
        # Try different formats for lookup
        if isic_code_clean in isic_dict:
            description = isic_dict.get(isic_code_clean)
        else:
            padded_code = isic_code_clean.zfill(4)
            if padded_code in isic_dict:
                description = isic_dict.get(padded_code)
            else:
                # As a fallback, try to find a close match
                for key in isic_dict.keys():
                    if isic_code_clean in key or key in isic_code_clean:
                        description = isic_dict.get(key)
                        break
                else:
                    description = "Description not found"
        
        alternatives.append({
            'code': isic_code,
            'confidence': confidence,
            'description': description
        })
    
    return top_isic_code, top_confidence, top_description, alternatives

# Speech to text function
def speech_to_text():
    try:
        r = sr.Recognizer()
        st.info("🎙️ Please speak your business description when ready...")
        
        # Create a button that will start the recording when clicked
        if "recording" not in st.session_state:
            st.session_state.recording = False
        
        col1, col2 = st.columns([1, 3])
        with col1:
            start_button = st.button("🎙️ Start Recording", key="start_recording")
        with col2:
            rec_status = st.empty()
        
        # Only show stop button if recording is active
        if st.session_state.recording:
            stop_button = st.button("⏹️ Stop Recording", key="stop_recording", type="primary")
        else:
            stop_button = False
        
        audio_placeholder = st.empty()
        
        if start_button:
            st.session_state.recording = True
            rec_status.warning("🔴 Recording... (speak now)")
            
            # Use microphone as source
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=10, phrase_time_limit=20)
                
                # Save audio to session state
                st.session_state.audio_data = audio.get_wav_data()
                
                # Display audio player
                audio_b64 = base64.b64encode(st.session_state.audio_data).decode()
                audio_placeholder.markdown(f"""
                    <audio controls>
                        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Stop recording
                st.session_state.recording = False
                rec_status.success("✅ Recording complete!")
                
                # Perform speech recognition
                try:
                    text = r.recognize_google(audio)
                    st.session_state.speech_text = text
                    return text
                except sr.UnknownValueError:
                    st.error("Could not understand audio. Please try again.")
                    return None
                except sr.RequestError:
                    st.error("Could not request results from speech recognition service.")
                    return None
        
        elif stop_button:
            st.session_state.recording = False
            rec_status.success("✅ Recording stopped.")
            return None
        
        # If we have speech text in session state, return it
        if "speech_text" in st.session_state:
            return st.session_state.speech_text
            
        return None
        
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return None

# CSS for the app
st.markdown("""
<style>
.main-container {
    padding: 2rem;
    border-radius: 10px;
    background-color: #f8f9fa;
    margin-bottom: 2rem;
}
.result-container {
    padding: 1.5rem;
    border-radius: 8px;
    background-color: #e9ecef;
    margin-top: 2rem;
}
.isic-code {
    font-size: 1.8rem;
    font-weight: bold;
    color: #0066cc;
}
.isic-description {
    font-size: 1.2rem;
    font-style: italic;
    color: #333;
    margin-top: 0.5rem;
}
.confidence {
    font-size: 1.2rem;
    color: #28a745;
}
.alt-container {
    padding: 1rem;
    border-radius: 8px;
    background-color: #f1f8ff;
    margin: 0.5rem 0;
    border-left: 4px solid #0066cc;
}
.alt-header {
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: #333;
}
.alt-code {
    font-weight: bold;
    color: #0066cc;
}
.alt-confidence {
    font-size: 0.9rem;
    color: #28a745;
}
.alt-description {
    font-size: 0.9rem;
    font-style: italic;
    color: #555;
    margin-top: 0.3rem;
}
.voice-btn {
    background-color: #ff4b4b;
    color: white;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    margin: 10px 0;
}
.voice-btn:hover {
    background-color: #ff7171;
}
.voice-recording {
    animation: pulse 1.5s infinite;
    background-color: #ff0000;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Main app function
def main():
    # Load models and data at startup
    try:
        with st.spinner("Loading BERT classification model and ISIC data..."):
            model, label_encoder, isic_dict = load_model_and_data("single")
        st.success("Model and ISIC data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.info("Please ensure that 'classes.npy', 'isic_perfect.csv' and the model directory are in the correct locations.")
        st.stop()
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Business Description")
        
        # Input method tabs
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "Voice Input"],
            horizontal=True
        )
        
        # Initialize text in session state if not present
        if "text_input" not in st.session_state:
            st.session_state.text_input = ""
        
        # Handle input method
        if input_method == "Text Input":
            description = st.text_area(
                "Describe the business activity in detail",
                value=st.session_state.text_input,
                height=150,
                placeholder="Example: Manufacturing of wooden furniture for household use"
            )
            # Store the description in session state
            st.session_state.text_input = description
        else:
            # Voice input method
            speech_text = speech_to_text()
            
            if speech_text:
                st.text_area(
                    "Transcribed text (edit if needed):",
                    value=speech_text,
                    key="transcribed_text",
                    height=150
                )
                description = st.session_state.transcribed_text
                # Also update text input for consistency
                st.session_state.text_input = description
            else:
                if "transcribed_text" in st.session_state:
                    description = st.session_state.transcribed_text
                else:
                    description = ""
                    
                # Show a text area for editing the transcribed text
                if description:
                    description = st.text_area(
                        "Transcribed text (edit if needed):",
                        value=description,
                        key="edit_transcribed",
                        height=150
                    )
        
        classify_button = st.button("Classify", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("Recent Classifications")
        if 'history' not in st.session_state:
            st.session_state.history = []
            
        # Display history (in reverse order - newest first)
        for item in reversed(st.session_state.history[-5:]):
            with st.container():
                st.markdown(f"""
                    <div style="padding: 0.5rem; border-bottom: 1px solid #ddd; margin-bottom: 0.5rem;">
                        <div style="font-size: 0.8rem; color: #666;">"{item['text'][:50]}{'...' if len(item['text']) > 50 else ''}"</div>
                        <div style="font-weight: bold; color: #0066cc;">{item['code']}</div>
                        <div style="font-size: 0.7rem; color: #28a745;">Confidence: {item['confidence']:.2%}</div>
                        <div style="font-size: 0.7rem; font-style: italic;">{item['description'][:70]}{'...' if len(item['description']) > 70 else ''}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Classification logic
    if classify_button and description:
        with st.spinner("Classifying..."):
            # Add a progress bar for user feedback
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get the prediction with alternatives
            isic_code, confidence, description_text, alternatives = classify_description(description, model, label_encoder, isic_dict)
            
            # Add to history (handle potential None or empty description_text)
            st.session_state.history.append({
                'text': description,
                'code': isic_code,
                'confidence': confidence,
                'description': description_text if description_text else "Description not available"
            })
        
        # Display the top result
        st.markdown("### Classification Result")
        st.markdown(f"""
            <div class="result-container">
                <p>The business description has been classified as:</p>
                <p class="isic-code">{isic_code}</p>
                <p class="isic-description">{description_text}</p>
                <p class="confidence">Confidence Score: {confidence:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display alternative predictions
        st.markdown("### Alternative Classifications")
        st.info("The model provides these alternative classifications, ranked by confidence. Consider these if the top prediction doesn't seem accurate.")
        
        alt_cols = st.columns(2)
        
        # Counter to track displayed alternatives
        displayed_alternatives = 0
        
        for i, alt in enumerate(alternatives):
            # Skip the first one if it's the same as the top prediction
            if i == 0 and alt['code'] == isic_code:
                continue
                
            # Only display up to 4 alternatives
            if displayed_alternatives >= 4:
                break
                
            with alt_cols[displayed_alternatives % 2]:
                st.markdown(f"""
                    <div style="padding: 1rem; margin-bottom: 0.5rem; border-radius: 5px; background-color: {'#e9f7ef' if displayed_alternatives < 2 else '#f5f5f5'};">
                        <div style="font-weight: bold; font-size: 1.2rem;">{alt['code']}</div>
                        <div style="color: #28a745;">Confidence: {alt['confidence']:.2%}</div>
                        <div class="alt-description">{alt['description']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            displayed_alternatives += 1
        
        # Show a more detailed explanation
        with st.expander("Explanation"):
            st.markdown(f"""
                * **ISIC Code**: {isic_code}
                * **Description**: {description_text}
                * **Confidence**: {confidence:.2%}
                * **Business Activity Analyzed**: "{description}"
                
                The BERT model analyzed the text description and determined the most likely 
                ISIC classifications, ranked by confidence score.
                
                Each ISIC code corresponds to a specific category of economic activity as defined
                in the International Standard Industrial Classification system.
                
                Consider reviewing the alternative classifications if the top prediction 
                doesn't seem to match the business activity described.
                
                Higher confidence scores (closer to 100%) indicate greater certainty in the prediction.
            """)
    
    # Display a message if no description is provided
    elif classify_button and not description:
        st.warning("Please enter a business description to classify.")

# Sidebar for additional information
with st.sidebar:
    st.subheader("About")
    st.write("""
        This tool helps classify industry descriptions according to the
        International Standard Industrial Classification (ISIC).
    """)
    
    st.subheader("Instructions")
    st.write("""
        1. Enter a detailed description of the business activity (type or speak)
        2. Click 'Classify' to get the ISIC code prediction with its official description
        3. View the result with confidence score
        4. Recent classifications will appear in the history panel
    """)
    
    st.subheader("Speech Recognition")
    st.write("""
        The voice input feature allows you to speak your business description:
        1. Select 'Voice Input' option
        2. Click 'Start Recording' and speak clearly
        3. Review and edit the transcribed text if needed
        4. Click 'Classify' to process
    """)
    
    st.subheader("Model Information")
    st.write("""
        This application uses a fine-tuned BERT model for industry classification.
        The model predicts ISIC codes based on text descriptions of economic activities.
        The descriptions for each ISIC code are sourced from the official ISIC classification.
    """)
    
    st.subheader("Batch Processing")
    st.write("""
        Need to classify multiple descriptions? Check out our CSV batch processing option:
    """)
    
    if st.button("Switch to Batch Mode"):
        st.session_state.mode = "batch"
        st.rerun()

# Add batch processing mode with the same BERT model
def batch_mode():
    st.title("ISIC Classification Tool - Batch Mode")
    
    # Load the same BERT model for batch processing
    try:
        with st.spinner("Loading BERT classification model and ISIC data..."):
            model, label_encoder, isic_dict = load_model_and_data("batch")
        st.success("Model and ISIC data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.info("Please ensure that 'classes.npy', 'isic_perfect.csv' and the model directory are in the correct locations.")
        st.stop()
    
    st.markdown("""
        Upload a CSV file with descriptions to classify in batch. The CSV file should have a column 
        named 'text' or 'Description' containing the business activity descriptions to classify.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file with flexible encoding handling
            encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("Could not read CSV file with any of the attempted encodings")
                return
            
            # Handle different column naming (either 'text' or 'Description')
            if 'text' in df.columns:
                text_column = 'text'
            elif 'Description' in df.columns:
                text_column = 'Description'
                # Rename to 'text' for compatibility
                df['text'] = df['Description']
                text_column = 'text'
            else:
                st.error("CSV file must contain a column named 'text' or 'Description'.")
                return
            
            # Fill NaN values
            df = df.fillna('')
            
            # Display the first few rows
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Process button
            if st.button("Process Batch", type="primary"):
                with st.spinner("Processing batch data..."):
                    # Prepare progress indicators
                    total_rows = len(df)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Get all text values
                    text_values = list(df[text_column].values)
                    
                    # Make batch predictions
                    try:
                        # Show intermediate progress
                        status_text.text("Preparing model...")
                        progress_bar.progress(10)
                        
                        # Make predictions - using the same BERT model
                        status_text.text("Running predictions...")
                        predictions, raw_outputs = model.predict(text_values)
                        progress_bar.progress(70)
                        
                        # Calculate confidence scores
                        status_text.text("Calculating confidence scores...")
                        scores = []
                        for i, pred in enumerate(predictions):
                            certs = softmax(raw_outputs[i])
                            cert = certs[pred]
                            scores.append(cert)
                        progress_bar.progress(80)
                        
                        # Convert predictions to labels
                        status_text.text("Converting to ISIC codes...")
                        labels = label_encoder.inverse_transform(predictions)
                        progress_bar.progress(90)
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df["AI_label"] = labels
                        results_df["confidence"] = scores
                        
                        # Add ISIC descriptions
                        status_text.text("Adding ISIC descriptions...")
                        descriptions = []
                        
                        for label in labels:
                            label_clean = str(label).strip()
                            
                            if label_clean in isic_dict:
                                descriptions.append(isic_dict.get(label_clean))
                            else:
                                # Try padding with zeros
                                padded_label = label_clean.zfill(4)
                                if padded_label in isic_dict:
                                    descriptions.append(isic_dict.get(padded_label))
                                else:
                                    # Fallback to find closest match
                                    description_found = False
                                    for key in isic_dict.keys():
                                        if label_clean in key or key in label_clean:
                                            descriptions.append(isic_dict.get(key))
                                            description_found = True
                                            break
                                    
                                    if not description_found:
                                        descriptions.append("Description not found")
                        
                        # Add descriptions to results
                        results_df["ISIC_Description"] = descriptions
                        
                        # Complete progress
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        # Display results
                        st.success("Batch processing complete!")
                        st.write("Classification Results:")
                        st.dataframe(results_df)
                        
                        # Create download link for CSV
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="isic_classification_results.csv", 
                            mime="text/csv"
                        )
                        
                        # Also create Excel download option
                        try:
                            import io
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                results_df.to_excel(writer, index=False, sheet_name='ISIC Classifications')
                            
                            output.seek(0)
                            st.download_button(
                                label="Download Results as Excel",
                                data=output,
                                file_name="isic_classification_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except ImportError:
                            st.info("Excel export requires xlsxwriter package. CSV export is still available.")
                        
                    except Exception as e:
                        st.error(f"Error during batch processing: {e}")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Button to switch back to single mode
    if st.button("Switch back to Single Mode"):
        st.session_state.mode = "single"
        st.rerun()

if __name__ == "__main__":
    if 'mode' not in st.session_state:
        st.session_state.mode = "single"
        
    if st.session_state.mode == "single":
        main()
    else:
        batch_mode()