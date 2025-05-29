# ISIC Classification Tool with RAG

A modular application for classifying business activities according to ISIC (International Standard Industrial Classification) codes using BERT models and RAG (Retrieval-Augmented Generation) for enhanced explanations.

## Features

- 🤖 **BERT-based Classification**: Advanced neural network model for accurate ISIC code prediction
- 🎙️ **Voice Input**: Speech-to-text functionality for hands-free operation
- 🌍 **Multilingual Support**: Automatic language detection and translation
- 📚 **RAG Integration**: Context-aware explanations using Ollama and vector search
- 📊 **Batch Processing**: Handle multiple classifications simultaneously
- 📈 **Confidence Scoring**: Alternative predictions with confidence levels
- 💾 **Export Results**: Download classified data as CSV

## Project Structure

```
isic-classification-tool/
│
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── README.md                 # Setup instructions
│
├── config/
│   └── settings.py           # Configuration settings
│
├── models/
│   └── loader.py             # Model loading functions
│
├── rag/
│   └── system.py             # RAG system implementation
│
├── utils/
│   ├── classification.py     # Classification utilities
│   ├── language.py           # Language processing
│   └── speech.py             # Speech recognition
│
├── ui/
│   └── layout.py             # UI layout and styling
│
├── modes/
│   ├── single_mode.py        # Single classification mode
│   └── batch_mode.py         # Batch processing mode
│
└── data/                     # Data files (not included)
    ├── classes.npy           # Label encoder classes
    ├── isic_gam.csv      # ISIC codes and descriptions
```

## Setup Instructions

### 1. Environment Setup

```bash
# Clone or download the application files
# Navigate to the project directory
cd isic-classification-tool

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Data & Model Setup

To run the application, you need both the **ISIC reference files** and a **trained BERT model**.  

---

### 📂 Reference Files  

Place the following files in the **data directory**:  

- `classes.npy` → Label encoder classes file, classes.npy will be automatically created
- `isic_gam.csv` → ISIC codes and descriptions, # use your oww

---

### 📊 Training Data  

Prepare your dataset inside the `data/` directory with these files:  

- `train.csv` → Training data  
- `test.csv` → Evaluation data  

Both must include the following columns:  

- **`text`** → Business activity description  
- **`label`** → Corresponding ISIC code  


## 🏋️ Training & Setup Guide  

To use the ISIC Classification Tool, you must first prepare your dataset, train a BERT model, and update the configuration.  

---

### 📂 Training Data  

Place your dataset in the `data/` directory with the following files:  

- `train.csv` → Training data  
- `test.csv` → Evaluation data  

Each file must contain at least two columns:  

| text                              | label |
|-----------------------------------|-------|
| Small shop selling clothes        | 4771  |
| Fishing activity in coastal area  | 0311  |
| Car repair garage                 | 4520  |

---

### 🏋️ Train the Model  

Run the training script:  

```bash
python isic_classifier.py



### 3. Optional: RAG Setup

For enhanced explanations with RAG:

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull a model**: `ollama pull llama3.2`
3. **Create ISIC manual directory**: 
   ```bash
   mkdir isic_manual
   ```
4. **Add PDF manual**: Place `ISIC_SUMMARY_MANUAL.pdf` in `isic_manual/`


## Running the Application

```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## Configuration

Modify `config/settings.py` to adjust:

- **Model paths**: Update `MODEL_CONFIG` for different model locations
- **RAG settings**: Change Ollama URL, model name, or embedding model
- **File paths**: Adjust paths for data files
- **UI settings**: Modify page configuration

## Usage

### Single Classification Mode

1. Choose input method (text or voice)
2. Enter or speak business description
3. Click "🚀 Classify"
4. View results with confidence scores and alternatives
5. Read AI-powered explanations 

### Batch Processing Mode

1. Upload CSV file with 'text' or 'Description' column
2. Click "🚀 Process Batch"
3. View classification summary and visualizations
4. Download results as CSV

## Key Benefits

- ✅ **Faster & consistent ISIC coding**
- ✅ **Reduces human error & workload** 
- ✅ **Supports decision-making** with confidence scores & explanations
- ✅ **Batch mode** for large datasets (surveys, censuses)
- ✅ **Multilingual support** for international collaboration

## Troubleshooting

### Model Loading Issues
- Ensure model files are in the correct directory
- Check file paths in `config/settings.py`
- Verify CUDA availability for GPU acceleration

### Audio Issues
- Check microphone permissions
- Ensure PyAudio is properly installed
- Test microphone in browser settings

### RAG System Issues
- Verify Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Ensure PDF manual exists in specified path

### CSV Encoding Issues
- Try different encodings if upload fails
- Ensure CSV has 'text' or 'Description' column
- Remove special characters if necessary

## Technical Requirements

- Python 3.8+ to 3.11  
- 12GB+ RAM (16GB+ recommended for batch processing)
- GPU (improves performance)
- Internet connection (for translation services)
- Microphone (for voice input)