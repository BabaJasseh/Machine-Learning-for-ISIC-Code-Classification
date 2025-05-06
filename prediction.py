#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax
 
import torch

#-----------------------------------------------------------------------

use_cuda = torch.cuda.is_available()
cuda_device = 0

model_name = ["bert","google-bert/bert-base-uncased"]

#-----------------------------------------------------------------------

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

########################################################################

eval_df = pd.read_csv('isicnew.csv', encoding='latin1').fillna(0)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True) # read prefitted

model = ClassificationModel(model_name[0],
        "experiments/good-sweep-49wtfpcwu/best_model", # Assuming that is the folder
        use_cuda=use_cuda,
        cuda_device=cuda_device)


predictions, raw_outputs = model.predict(list(eval_df["text"].values))



# If scores are needed:
# softmax_scores = torch.nn.functional.softmax(torch.tensor(raw_outputs),dim=-1)
# scores = softmax_scores[:,predictions].detach().numpy() #how easy for the model to do prediction

scores = []
for i,pred in enumerate(predictions):
    certs = softmax(raw_outputs[i])
    cert = certs[pred]
    scores.append(cert)
eval_df["scores"] = scores


labels = label_encoder.inverse_transform(predictions)
eval_df["AI_label"] = labels



eval_df.to_csv("eval_predictions.csv",index=False)

# // shallow n n