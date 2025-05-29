import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# SimpleTransformers & WandB
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import wandb
import torch

# -----------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------

# Define a centralized configuration dictionary
CONFIG = {
    # Data Paths
    "DATA_DIR": Path("./data"),
    "TRAIN_FILE": "train.csv",
    "TEST_FILE": "test.csv",  
    "CLASSES_FILE": "classes.npy",
    "TEMP_DIR": Path("./temp_data"),

    # Model Configuration
    "MODEL_TYPE": "bert",
    "MODEL_NAME": "google-bert/bert-base-uncased",
    "NUM_LABELS": None,  
    "TEST_SIZE": 0.15,   
    "RANDOM_STATE": 42,  

    # WandB & Project 
    "PROJECT_NAME": "ISIC_CODE_CLASSIFICATION",
    
    # --- Sweep Configuration ---
    "SWEEP_CONFIG": {
        "method": "random",
        "metric": {"name": "weighted_f1", "goal": "maximize"}, 
        "parameters": {
                "train_batch_size": {"min": 8, "max": 64}, 
                "num_train_epochs": {"min": 3, "max": 8},
                "learning_rate": {"min": 1e-6, "max": 5e-5},  
        },
        "early_terminate": {"type": "hyperband", "min_iter": 3},
    },


    # --- Hardware & Environment ---
    "USE_CUDA": torch.cuda.is_available(),
    "CUDA_DEVICE": 0,
}

os.environ['WANDB_API_KEY'] = "YOUR_WANDB_API_KEY" 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

CONFIG["DATA_DIR"].mkdir(exist_ok=True)
CONFIG["TEMP_DIR"].mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# 2. LOGGING SETUP
# -----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
my_logger = logging.getLogger(CONFIG["PROJECT_NAME"])


# -----------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------

def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the F1-score, weighted by the support of each class.
    This is preferred over simple accuracy for imbalanced multiclass problems.
    """
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def compute_class_weights(train_df: pd.DataFrame, label_encoder: LabelEncoder) -> List[float]:
    """
    Computes 'balanced' class weights for the loss function and maps them
    back to the full set of all possible class indices.
    """
    unique_labels_in_train = np.unique(train_df['labels'])
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels_in_train,
        y=train_df['labels']
    )
    
    weight_map = {label: weight for label, weight in zip(unique_labels_in_train, weights)}
    
    weight_list = [weight_map.get(i, 1.0) 
                   for i in range(len(label_encoder.classes_))]
    
    my_logger.info(f"Class weights computed for {len(unique_labels_in_train)}/{len(label_encoder.classes_)} classes in training set.")
    return weight_list

def prefit_encoder(data_dir: Path, train_file: str, test_file: str, classes_file: str):
    """
    Loads training and final test data to fit a comprehensive LabelEncoder
    on all known classes to prevent label mismatch errors during deployment.
    Saves the class list to a file.
    """
    my_logger.info("--- Pre-fitting encoder on full dataset... ---")
    try:
        train_path = data_dir / train_file
        test_path = data_dir / test_file

    
        train_df = pd.read_csv(train_path, encoding='latin1').fillna("") 
        test_df = pd.read_csv(test_path, encoding='latin1').fillna("")
        
        full_df = pd.concat([train_df['label'], test_df['label']], ignore_index=True).astype(str).unique()

        label_encoder = LabelEncoder()
        label_encoder.fit(full_df)
        np.save(data_dir / classes_file, label_encoder.classes_)
        CONFIG["NUM_LABELS"] = len(label_encoder.classes_)
        my_logger.info(f"Total unique classes found and saved: {CONFIG['NUM_LABELS']}")

    except FileNotFoundError as e:
        my_logger.error(f"Data file not found: {e}. Ensure data files are in the {data_dir} directory.")
        sys.exit(1)
    except Exception as e:
        my_logger.error(f"Error during pre-fitting: {e}")
        sys.exit(1)


# -----------------------------------------------------------------------
# 4. TRAINING FUNCTION
# -----------------------------------------------------------------------

def train() -> int:
    """
    Main training loop for one hyperparameter run, managed by WandB.
    This function implements the fix for single-sample stratification errors.
    """
    with wandb.init() as run:
        run_id = run.name + "-" + run.id
        my_logger.info(f"Starting run: {run_id}")

        # --- Load Data ---
        train_val_df = pd.read_csv(CONFIG["DATA_DIR"] / CONFIG["TRAIN_FILE"], encoding='latin1').fillna("")
        
        label_counts = train_val_df['label'].value_counts()

        single_sample_labels = label_counts[label_counts < 2].index.tolist()
        
        if single_sample_labels:
            my_logger.warning(
                f"Removing {len(single_sample_labels)} single-sample classes before stratified split. "
                f"Total samples removed: {len(single_sample_labels)}"
            )
            train_val_df_filtered = train_val_df[~train_val_df['label'].isin(single_sample_labels)]
        else:
            train_val_df_filtered = train_val_df

     
        train_df, eval_df = train_test_split(
            train_val_df_filtered, 
            test_size=CONFIG["TEST_SIZE"], 
            random_state=CONFIG["RANDOM_STATE"],
            stratify=train_val_df_filtered['label'] 
        )


        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(CONFIG["DATA_DIR"] / CONFIG["CLASSES_FILE"], allow_pickle=True)
        
        train_df["labels"] = label_encoder.transform(train_df['label'].astype(str))
        eval_df["labels"] = label_encoder.transform(eval_df['label'].astype(str))

        class_weights = compute_class_weights(train_df, label_encoder)
        weight_list = [float(w) for w in class_weights]
        
        train_df = train_df[["text", "labels"]]
        eval_df = eval_df[["text", "labels"]]


        model_args = ClassificationArgs()
        model_args.learning_rate = wandb.config.learning_rate
        model_args.num_train_epochs = wandb.config.num_train_epochs
        model_args.train_batch_size = wandb.config.train_batch_size
        
        steps_per_epoch = int(len(train_df) / model_args.train_batch_size)
        model_args.evaluate_during_training_steps = max(1, steps_per_epoch // 5) 
        
        model_args.evaluate_during_training = True
        model_args.output_dir = str(CONFIG["TEMP_DIR"] / "experiments" / run_id)
        model_args.best_model_dir = str(CONFIG["TEMP_DIR"] / "experiments" / run_id / "best_model")
        model_args.wandb_project = CONFIG["PROJECT_NAME"]
        
        model_args.reprocess_input_data = False
        model_args.do_lower_case = True
        model_args.no_cache = True
        
        my_logger.info(f"Eval steps (approx. 5x per epoch): {model_args.evaluate_during_training_steps}")
        my_logger.info(f"Model Args: {vars(model_args)}") 

        
        if CONFIG["NUM_LABELS"] is None:
            classes = np.load(CONFIG["DATA_DIR"] / CONFIG["CLASSES_FILE"], allow_pickle=True)
            CONFIG["NUM_LABELS"] = len(classes)

        model = ClassificationModel(
            model_type=CONFIG["MODEL_TYPE"],
            model_name=CONFIG["MODEL_NAME"],
            num_labels=CONFIG["NUM_LABELS"],
            args=model_args,
            use_cuda=CONFIG["USE_CUDA"],
            cuda_device=CONFIG["CUDA_DEVICE"],
            weight=weight_list 
        )

        # Define metrics
        metrics = {
            "acc": accuracy_score,
            "weighted_f1": weighted_f1_score
        }
        
        # Train model
        model.train_model(train_df, eval_df=eval_df, **metrics)

        # --- Log Final Results ---
        my_logger.info(f"Run {run_id} finished. Best model saved to {model_args.best_model_dir}")

    wandb.join()
    return 0

# -----------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------

if __name__ == "__main__":

    if not (CONFIG["DATA_DIR"] / CONFIG["CLASSES_FILE"]).exists() or CONFIG["NUM_LABELS"] is None:
        my_logger.warning("Classes file not found or NUM_LABELS not set. Running pre-fitting.")
        prefit_encoder(CONFIG["DATA_DIR"], CONFIG["TRAIN_FILE"], CONFIG["TEST_FILE"], CONFIG["CLASSES_FILE"])
    
    if CONFIG["NUM_LABELS"] is None:
        try:
             classes = np.load(CONFIG["DATA_DIR"] / CONFIG["CLASSES_FILE"], allow_pickle=True)
             CONFIG["NUM_LABELS"] = len(classes)
        except Exception:
             my_logger.error("Failed to load classes.npy after pre-fitting. Aborting.")
             sys.exit(1)

    try:
        my_logger.info(f"Starting WandB sweep for project: {CONFIG['PROJECT_NAME']}")
        sweep_id = wandb.sweep(CONFIG["SWEEP_CONFIG"], project=CONFIG["PROJECT_NAME"])

        wandb.agent(sweep_id, train, count=5)
    except Exception as e:
        my_logger.critical(f"A critical error occurred during the sweep execution: {e}")
        sys.exit(1)

    my_logger.info("--- All sweep runs completed successfully. ---")
