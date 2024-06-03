import os
from datamodule import Sec_Data_Module
from config import config
import pickle
from model import Sec_Classifier
import pytorch_lightning as pl
import torch
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["TOKENIZERS_PARALLELISM"] = "False"

model_list = ['xlm-roberta-base', 'bert-base-uncased', 'jackaduma/SecRoBERTa']
for model_name in model_list:
    config['model_name'] = model_name
    epoch_num = config['n_epochs']

    print(f"{epoch_num}epoch train started!")

    model_name = config['model_name'].split('/')[-1]
    data_path = f'../{model_name}_dataset/'
    train_path = data_path + 'train.pkl'
    val_path = data_path + 'val.pkl'

    data_module = Sec_Data_Module(train_path, val_path, batch_size=config['batch_size'], model_name=config['model_name'], max_token_len=512)
    data_module.setup()
    config['train_size'] = len(data_module.train_dataloader())
    model = Sec_Classifier(config)

    trainer = pl.Trainer(logger=False, max_epochs=config['n_epochs'], num_sanity_val_steps=-1, accelerator="gpu", devices=1)
    trainer.fit(model, data_module)

    if not os.path.exists(f'{model_name}_ckpt'):
        os.mkdir(f'{model_name}_ckpt')

    model_save_dir = f'./{model_name}_ckpt/'



    print(f"{epoch_num}epoch train completed!")
