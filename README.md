# Machine-Learning-for-ISIC-Code-Classification
Automated ISIC Code Classification using BERT. 
# ISIC Code Classification using BERT

This repository contains a practical implementation of an automated ISIC (International Standard Industrial Classification) code classifier using BERT. Developed by the IT Unit of the Gambia Bureau of Statistics (GBoS), this project aims to modernize the classification of economic activity descriptions for statistical reporting.

## 🔍 Project Overview

Manual ISIC classification is often time-consuming, inconsistent, and difficult to scale. This system automates the process using a fine-tuned BERT model that achieves over **93% accuracy** on real business registration data.

### Features:
- BERT-based classification using PyTorch
- Streamlit web interface for easy use
- Inference pipeline with confidence scores
- Visualization of classification results
- Export of classified data (CSV)
- Integration-ready for statistical systems

## 🧠 How It Works

1. **Training**:
   - Preprocess business descriptions
   - Encode ISIC labels
   - Fine-tune pre-trained BERT using PyTorch

2. **Inference**:
   - Upload new descriptions via Streamlit app
   - Model predicts ISIC codes with confidence scores
   - Output includes visualization and CSV download

3. **User Interface**:
   - Simple file upload (CSV)
   - Realtime classification
   - Results preview + chart display

## 📊 Performance

| Metric        | Result       |
|---------------|--------------|
| Accuracy      | 93%+         |
| Processing    | Seconds/entry |
| Throughput    | ~25,000/day  |
| Staff Time Saved | 85%+     |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- Streamlit
- Pandas, NumPy, scikit-learn

### Setup

```bash
# Clone the repo
git clone https://github.com/GBoS-IT/isic-classification.git
cd isic-classification

# Install dependencies
pip install -r requirements.txt
