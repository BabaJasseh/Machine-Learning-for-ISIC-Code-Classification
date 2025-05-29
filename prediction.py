import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax
import torch
import os


def main():
    # --------------------
    # CONFIG
    # --------------------
    use_cuda = torch.cuda.is_available()
    cuda_device = 0
    model_type = "bert"
    model_path = "temp_data/experiments/celestial-sweep-1-lszxepju/best_model"  # trained model
    classes_file = "data/classes.npy"
    data_file = "isicnew.csv"  # Can be labeled or unlabeled

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --------------------
    # LOAD DATA
    # --------------------
    df = pd.read_csv(data_file, encoding='latin1').fillna("")

    # --------------------
    # LOAD ENCODER & MODEL
    # --------------------
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(classes_file, allow_pickle=True)

    model = ClassificationModel(
        model_type,
        model_path,
        use_cuda=use_cuda,
        cuda_device=cuda_device
    )

    # --------------------
    # PREDICT
    # --------------------
    predictions, raw_outputs = model.predict(list(df["text"].values))

    # Confidence scores
    scores = [softmax(raw_outputs[i])[pred] for i, pred in enumerate(predictions)]
    df["AI_score"] = scores

    # Map back to labels
    df["AI_label"] = label_encoder.inverse_transform(predictions)

    # --------------------
    # IF TRUE LABELS EXIST (TESTING)
    # --------------------
    if "label" in df.columns:
        df["label_num"] = label_encoder.transform(df["label"].astype(str))
        acc = accuracy_score(df["label_num"], predictions)
        f1 = f1_score(df["label_num"], predictions, average="weighted")
        print(f"Test Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")

    # --------------------
    # SAVE OUTPUT
    # --------------------
    df.to_csv("predictions_output.csv", index=False)


if __name__ == "__main__":
    main()
