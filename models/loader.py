"""
Model loading and initialization functions
"""

import pandas as pd
import numpy as np
import streamlit as st
import torch
from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel
from config.settings import MODEL_CONFIG, FILE_PATHS, ENCODINGS

@st.cache_resource
def load_model_and_data(mode="single"):
    """Load BERT model, label encoder, and ISIC data"""
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    cuda_device = 0 if use_cuda else -1
    
    # Load the label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(FILE_PATHS["classes"], allow_pickle=True)
    
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
        for encoding in ENCODINGS:
            try:
                isic_df = pd.read_csv(FILE_PATHS["isic_csv"], encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise Exception("Could not read CSV file with any of the attempted encodings")
        
        isic_df['CLASS'] = isic_df['CLASS'].astype(str).str.strip()
        isic_dict = dict(zip(isic_df['CLASS'], isic_df['ECONOMICS ACTIVITIES']))
        
    except Exception as e:
        st.error(f"Error loading ISIC data: {e}")
        isic_dict = {}
    
    return model, label_encoder, isic_dict