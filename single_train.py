#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import sklearn
import numpy as np

from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
#-----------------------------------------------------------------------

use_cuda = torch.cuda.is_available()
cuda_device = 0

model_name = ["bert","google-bert/bert-base-uncased"]

#-----------------------------------------------------------------------

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_name = "ISIC CODE CLASSIFICATION"

#-----------------------------------------------------------------------

import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
my_logger = logging.getLogger(project_name)

########################################################################

def prefit_encoder():
    my_logger.info("  Preffiting encoder")

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    full_df = pd.concat([train_df,test_df],ignore_index=True).fillna(0)
    label_codificador = LabelEncoder() # transform the isic code in incremental, the model expects to recieve

    # Assuminb "label" as label column
    label_codificador.fit(full_df["label"]) # ie the output the one the model is going to predict
    np.save('classes.npy', label_codificador.classes_)

def train(args):
    # Add needed arguments
    model_args = ClassificationArgs()
    model_args.learning_rate=args.learning_rate
    model_args.num_train_epochs=args.num_train_epochs
    model_args.train_batch_size = args.train_batch_size
    model_args.reprocess_input_data = False
    model_args.evaluate_during_training = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.overwrite_output_dir = True
    model_args.warmup_ratio = 0.06
    model_args.do_lower_case = True

    eval_split = 5

    train_df = pd.read_csv('train.csv').fillna(0)
    eval_df = pd.read_csv('test.csv').fillna(0)

    train_df["label_original"] = train_df["label"]
    eval_df["label_original"] = eval_df["label"]

    label_codificador = LabelEncoder()
    label_codificador.classes_ = np.load('classes.npy', allow_pickle=True) # read prefitted

    train_df["labels"] = label_codificador.transform(train_df['label_original'])
    eval_df["labels"] = label_codificador.transform(eval_df["label_original"])

    # Only text and label columns are relevant
    train_df = train_df[["text","labels"]]
    eval_df = eval_df[["text","labels"]]

    model_args.evaluate_during_training_steps = int(len(train_df)/(eval_split*model_args.train_batch_size))
    my_logger.info("  Eval steps: {}".format(model_args.evaluate_during_training_steps))

    model = ClassificationModel(model_name[0],
            model_name[1],
            args=model_args, # controled by wandb
            num_labels=len(label_codificador.classes_),
            use_cuda=use_cuda,
            cuda_device=cuda_device)

    model.train_model(train_df,acc=sklearn.metrics.accuracy_score,eval_df=eval_df)

    return 0

#---

if __name__ == '__main__':
    import sys
    # Add needed arguments
    parser = argparse.ArgumentParser(description="Trains model given learning_rate, num_train_epochs and train_batch_size")
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size',type=int,default=8)
    args = parser.parse_args()

    if not Path("classes.npy").exists():
        prefit_encoder()

    sys.exit(train(args))
