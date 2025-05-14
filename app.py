#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
import os
import time
import base64
import torch
from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax

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
    Upload a CSV file containing descriptions, and the model will predict 
    the appropriate classification codes.
""")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to load models
@st.cache_resource
def load_model():
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    cuda_device = 0 if use_cuda else -1
    
    # Load the label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    
    # Load the model
    model = ClassificationModel(
        "bert",
        "experiments/good-sweep-49wtfpcwu/best_model",
        use_cuda=use_cuda,
        cuda_device=cuda_device
    )
    
    return model, label_encoder

# Function to create a download link for the dataframe
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

# Function to process the uploaded file
def process_file(df, model, label_encoder):
    # Ensure the text column exists
    if "text" not in df.columns:
        # Try to find a suitable text column
        text_columns = [col for col in df.columns if 
                        any(keyword in col.lower() for keyword in 
                           ["description", "text", "activity", "occupation", "industry"])]
        
        if text_columns:
            st.info(f"Using '{text_columns[0]}' as the text column for classification.")
            df["text"] = df[text_columns[0]]
        else:
            st.error("Could not find a suitable text column. Please ensure your CSV contains a 'text' column or a column with description data.")
            return None
    
    # Make predictions
    with st.spinner("Running BERT model for predictions..."):
        predictions, raw_outputs = model.predict(list(df["text"].values))
    
    # Calculate confidence scores
    scores = []
    for i, pred in enumerate(predictions):
        certs = softmax(raw_outputs[i])
        cert = certs[pred]
        scores.append(cert)
    
    # Convert predicted indices to actual labels
    labels = label_encoder.inverse_transform(predictions)
    
    # Add predictions to dataframe
    df["AI_label"] = labels
    df["confidence_score"] = scores
    
    return df

# CSS for download button
st.markdown("""
<style>
.download-button {
    display: inline-block;
    padding: 0.5em 1em;
    background-color: #4CAF50;
    color: white;
    text-align: center;
    text-decoration: none;
    font-size: 16px;
    border-radius: 4px;
    transition: background-color 0.3s;
}
.download-button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Main app function
def main():
    # Load models at startup
    try:
        with st.spinner("Loading BERT classification model..."):
            model, label_encoder = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure that 'classes.npy' and the model directory are in the correct locations.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file with descriptions", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file).fillna("")
            
            # Show file preview
            st.subheader("Preview of uploaded data")
            st.dataframe(df.head())
            
            # Process button
            if st.button("Run Classification"):
                with st.spinner("Processing... This may take a few minutes depending on the file size."):
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    
                    # Simulate progress while processing
                    for percent_complete in range(100):
                        time.sleep(0.05)  # Adjust time for realistic progress simulation
                        progress_bar.progress(percent_complete + 1)
                    
                    # Actual processing
                    result_df = process_file(df, model, label_encoder)
                
                if result_df is not None:
                    st.success("Classification completed!")
                    
                    # Show preview of results
                    st.subheader("Preview of classification results")
                    st.dataframe(result_df.head())
                    
                    # Provide download link
                    st.markdown(
                        get_download_link(result_df, "classified_data.csv", "📥 Download Classified Data (CSV)"),
                        unsafe_allow_html=True
                    )
                    
                    # Show distribution of predicted labels
                    st.subheader("Classification Distribution")
                    label_counts = result_df['AI_label'].value_counts().head(15)
                    st.bar_chart(label_counts)
                    
                    # Show confidence score statistics
                    st.subheader("Confidence Score Statistics")
                    avg_score = result_df['confidence_score'].mean()
                    min_score = result_df['confidence_score'].min()
                    max_score = result_df['confidence_score'].max()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Confidence", f"{avg_score:.2%}")
                    col2.metric("Minimum Confidence", f"{min_score:.2%}")
                    col3.metric("Maximum Confidence", f"{max_score:.2%}")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please check the file format and contents.")

# Sidebar for additional information
with st.sidebar:
    st.subheader("About")
    st.write("""
        This tool helps classify industry descriptions according to the
        International Standard Industrial Classification (ISIC).
    """)
    
    st.subheader("Instructions")
    st.write("""
        1. Prepare a CSV file with a column named 'text' containing industry descriptions
        2. Upload the file using the button on the main panel
        3. Click 'Run Classification' to process the data
        4. Download the results after processing completes
    """)
    
    st.subheader("Model Information")
    st.write("""
        This application uses a fine-tuned BERT model for industry classification.
        The model predicts ISIC codes based on text descriptions of economic activities.
    """)
    
    st.subheader("Required Files")
    st.write("""
        Make sure these files are in the same directory as this app:
        - `classes.npy`: Label encoder classes
        - `experiments/vocal-sweep-4495v5pww/best_model/`: BERT model directory
    """)

if __name__ == "__main__":
    main()