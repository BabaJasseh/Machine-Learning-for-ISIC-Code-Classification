#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import sklearn
import numpy as np

from sklearn.preprocessing import LabelEncoder
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import wandb
import torch
#-----------------------------------------------------------------------

use_cuda = torch.cuda.is_available()
cuda_device = 0

model_name = ["bert","google-bert/bert-base-uncased"]

#-----------------------------------------------------------------------

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_API_KEY'] = "cbf883ba06ced3977aaeb3c8753d7dbe9e923fbe"

#-----------------------------------------------------------------------

project_name = "ISIC CODE CLASSIFICATION"

sweep_config = {
    "method": "random",  # bayes, grid, random
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "train_batch_size":{"min": 8,"max":256},
        "num_train_epochs": {"min": 3,"max":10},
        "learning_rate": {"min": 1e-9, "max": 1e-3},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
}

#-----------------------------------------------------------------------

import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
my_logger = logging.getLogger(project_name)

########################################################################

def prefit_encoder():
    my_logger.info("Preffiting encoder")

    train_df = pd.read_csv('train.csv', encoding='latin1')
    test_df = pd.read_csv('test.csv', encoding='latin1')

    full_df = pd.concat([train_df,test_df],ignore_index=True).fillna(0)
    label_codificador = LabelEncoder() # transform the isic code in incremental, the model expects to recieve

    # Assuminb "label" as label column
    label_codificador.fit(full_df["label"]) # ie the output the one the model is going to predict
    np.save('classes.npy', label_codificador.classes_)

def train():  # the experiment, connecting to the server etc. call by wandb
    with wandb.init() as run:
        run_id = run.name+run.id
        # Add needed arguments
        model_args = ClassificationArgs()
        model_args.learning_rate=wandb.config.learning_rate
        model_args.num_train_epochs=wandb.config.num_train_epochs
        model_args.train_batch_size = wandb.config.train_batch_size
        model_args.reprocess_input_data = False
        model_args.evaluate_during_training = True
        model_args.use_multiprocessing = False
        model_args.use_multiprocessing_for_evaluation = False
        model_args.warmup_ratio = 0.06
        model_args.output_dir = "experiments/"+run_id
        model_args.best_model_dir = "experiments/"+run_id+"/best_model"
        model_args.do_lower_case = True
        model_args.wandb_project = project_name

        eval_split = 5

        train_df = pd.read_csv('train.csv', encoding='latin1')
        eval_df = pd.read_csv('test.csv' , encoding='latin1')

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

    wandb.join()
    return 0

#---

if not Path("classes.npy").exists():
    prefit_encoder()

sweep_id = wandb.sweep(sweep_config, project=project_name)
wandb.agent(sweep_id,train,count=5)
