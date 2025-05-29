"""
Batch processing mode for multiple classifications
"""

import pandas as pd
import streamlit as st
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.special import softmax
from models.loader import load_model_and_data
from config.settings import ENCODINGS
from utils.classification import get_isic_description

def batch_processing_mode():
    """Main function for batch processing mode"""
    st.title("📊 ISIC Classification Tool - Batch Mode")
    
    # Load the same BERT model for batch processing
    try:
        with st.spinner("Loading BERT classification model and ISIC data..."):
            model, label_encoder, isic_dict = load_model_and_data("batch")
        st.success("✅ Model and ISIC data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.info("Please ensure that 'classes.npy', 'isic_gam.csv' and the model directory are in the correct locations.")
        st.stop()
    
    st.markdown("""
        📁 Upload a CSV file with descriptions to classify in batch. The CSV file should have a column 
        named 'text' or 'Description' containing the business activity descriptions to classify.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        process_batch_file(uploaded_file, model, label_encoder, isic_dict)

def process_batch_file(uploaded_file, model, label_encoder, isic_dict):
    """Process the uploaded batch file"""
    try:
        # Read the CSV file with flexible encoding handling
        df = read_csv_with_encoding(uploaded_file)
        if df is None:
            return
        
        # Handle different column naming
        df = handle_column_names(df)
        if df is None:
            return
        
        # Fill NaN values
        df = df.fillna('')
        
        # Display the first few rows
        st.write("👀 Preview of uploaded data:")
        st.dataframe(df.head())
        
        # Process button
        if st.button("🚀 Process Batch", type="primary"):
            process_batch_predictions(df, model, label_encoder, isic_dict)
            
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

def read_csv_with_encoding(uploaded_file):
    """Read CSV file with multiple encoding attempts"""
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.success(f"✅ Successfully loaded CSV with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
    
    st.error("❌ Could not read CSV file with any of the attempted encodings")
    return None

def handle_column_names(df):
    """Handle different column naming conventions"""
    if 'text' in df.columns:
        text_column = 'text'
    elif 'Description' in df.columns:
        text_column = 'Description'
        # Rename to 'text' for compatibility
        df['text'] = df['Description']
        text_column = 'text'
    else:
        st.error("❌ CSV file must contain a column named 'text' or 'Description'.")
        return None
    
    return df

def process_batch_predictions(df, model, label_encoder, isic_dict):
    """Process batch predictions and display results"""
    with st.spinner("🔄 Processing batch data..."):
        # Prepare progress indicators
        total_rows = len(df)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get all text values
        text_values = list(df['text'].values)
        
        # Make batch predictions
        try:
            # Show intermediate progress
            status_text.text("🔄 Preparing model...")
            progress_bar.progress(10)
            
            # Make predictions - using the same BERT model
            status_text.text("🧠 Running predictions...")
            predictions, raw_outputs = model.predict(text_values)
            progress_bar.progress(70)
            
            # Calculate confidence scores and create results
            results_df = create_results_dataframe(
                df, predictions, raw_outputs, label_encoder, 
                isic_dict, progress_bar, status_text
            )
            
            # Display results and visualizations
            display_batch_results(results_df, total_rows)
            
        except Exception as e:
            st.error(f"Error during batch processing: {e}")

def create_results_dataframe(df, predictions, raw_outputs, label_encoder, isic_dict, progress_bar, status_text):
    """Create results dataframe with predictions and confidence scores"""
    # Calculate confidence scores
    status_text.text("📊 Calculating confidence scores...")
    scores = []
    for i, pred in enumerate(predictions):
        certs = softmax(raw_outputs[i])
        cert = certs[pred]
        scores.append(cert)
    progress_bar.progress(80)
    
    # Convert predictions to labels
    status_text.text("🏷️ Converting to ISIC codes...")
    labels = label_encoder.inverse_transform(predictions)
    progress_bar.progress(90)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['ISIC_Code'] = labels
    results_df['Confidence'] = scores
    
    # Map ISIC descriptions
    descriptions = []
    for code in labels:
        code_clean = str(code).strip()
        description = get_isic_description(code_clean, isic_dict)
        descriptions.append(description)
    
    results_df['ISIC_Description'] = descriptions
    progress_bar.progress(100)
    status_text.success("✅ Batch classification complete!")
    
    return results_df

def display_batch_results(results_df, total_rows):
    """Display batch processing results and visualizations"""
    # Display sample results
    st.write("📄 Sample of classified results:")
    st.dataframe(results_df.head())

    # Summary section
    st.subheader("📊 Batch Classification Summary")

    # Summary metrics
    st.metric("Total rows processed", total_rows)
    st.metric("Average confidence", f"{results_df['Confidence'].mean():.2%}")

    # Most common ISIC codes
    most_common = results_df['ISIC_Code'].value_counts().head(5)
    st.write("Most common ISIC codes in this batch:")
    st.bar_chart(most_common)

    # Create visualizations
    create_batch_visualizations(results_df)
    
    # Create download link
    create_download_link(results_df)

def create_batch_visualizations(results_df):
    """Create visualizations for batch results"""
    # Confidence distribution histogram
    st.subheader("📈 Confidence Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(results_df['Confidence'], bins=20, color="#667eea", edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of Predictions")
    ax.set_title("Distribution of Classification Confidence")
    st.pyplot(fig)

    # Top 10 ISIC codes frequency
    st.subheader("📊 Top 10 ISIC Codes Frequency")
    top_codes = results_df['ISIC_Code'].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    top_codes.plot(kind='bar', color="#38ef7d", ax=ax2)
    ax2.set_xlabel("ISIC Code")
    ax2.set_ylabel("Count")
    ax2.set_title("Top 10 ISIC Codes in Batch")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

def create_download_link(results_df):
    """Create download link for results"""
    csv_buffer = BytesIO()
    results_df.to_csv(csv_buffer, index=False)
    b64 = base64.b64encode(csv_buffer.getvalue()).decode()
    st.markdown(f"""
        <a href="data:file/csv;base64,{b64}" download="batch_classification_results.csv">
            📥 Download Classified CSV
        </a>
    """, unsafe_allow_html=True)