"""
Classification utility functions
"""

import numpy as np
from scipy.special import softmax

def classify_description(text, model, label_encoder, isic_dict):
    """Classify a single description with alternatives"""
    prediction, raw_output = model.predict([text])
    scores = softmax(raw_output[0])
    
    top_pred_idx = prediction[0]
    top_confidence = scores[top_pred_idx]
    top_isic_code = label_encoder.inverse_transform([top_pred_idx])[0]
    
    top_isic_code_clean = str(top_isic_code).strip()
    
    # Get description for top prediction
    top_description = get_isic_description(top_isic_code_clean, isic_dict)
    
    # Get top 5 alternatives
    top_indices = np.argsort(scores)[::-1][:5]
    alternatives = []
    
    for idx in top_indices:
        isic_code = label_encoder.inverse_transform([idx])[0]
        confidence = scores[idx]
        isic_code_clean = str(isic_code).strip()
        description = get_isic_description(isic_code_clean, isic_dict)
        
        alternatives.append({
            'code': isic_code,
            'confidence': confidence,
            'description': description
        })
    
    return top_isic_code, top_confidence, top_description, alternatives

def get_isic_description(isic_code_clean, isic_dict):
    """Get ISIC description for a given code"""
    if isic_code_clean in isic_dict:
        return isic_dict.get(isic_code_clean)
    else:
        padded_code = isic_code_clean.zfill(4)
        if padded_code in isic_dict:
            return isic_dict.get(padded_code)
        else:
            for key in isic_dict.keys():
                if isic_code_clean in key or key in isic_code_clean:
                    return isic_dict.get(key)
            return "Description not found"