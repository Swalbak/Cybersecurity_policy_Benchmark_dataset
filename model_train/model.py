import os
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import math
from torchmetrics import Accuracy, F1Score
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import wandb
import time
import numpy as np

class Sec_Classifier(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
        self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])

        # initialize
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout()
        self.model_name = config['model_name'].split('/')[-1]
        wandb.init(project=f'sec_classification_{self.model_name}', config=self.config)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # start_time = time.time()

        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # elapsed_time = time.time() - start_time
        # print(f'Elapsed time for one batch: {elapsed_time:.4f} seconds')

        loss = 0
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels)

        return loss, logits
    
    def training_step(self, batch, batch_index):
        # forword(input ids, attention_mask)에 batch들을 넘겨줌(model 인스턴스 호출 -> forword 메소드 호출)
        loss, outputs = self(**batch)
        self.log("train loss ", loss, prog_bar = True, logger=True)
        wandb.log({"train_loss": loss})
        self.train_step_outputs.append(outputs)
	
        return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        self.log("validation loss ", loss, prog_bar = True, logger=True)
        self.validation_step_outputs.append(outputs)
        wandb.log({"val_loss": loss})

        return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}

    def predict_step(self, batch, batch_index):
        loss, outputs = self(**batch)

        return {'loss': loss, 'outputs': outputs}
    
    def on_train_epoch_end(self):
        ckpt_dir = f'../model_ckpt/{self.model_name}_ckpt'
        predict_dir = f'../model_predictions/{self.model_name}_predictions'

        if not os.path.exists(os.path.join(predict_dir, 'train_predictions')):
            os.makedirs(os.path.join(predict_dir, 'train_predictions'), exist_ok=True)
        predict_dir = os.path.join(predict_dir, 'train_predictions')

        torch.save(self.train_step_outputs, os.path.join(predict_dir, f'{self.current_epoch}train.pkl'))
        self.train_step_outputs.clear()

        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        

    def on_validation_epoch_end(self):
        predict_dir = f'../model_predictions/{self.model_name}_predictions'

        if not os.path.exists(os.path.join(predict_dir, 'val_predictions')):
            os.makedirs(os.path.join(predict_dir, 'val_predictions'), exist_ok=True)
        predict_dir = os.path.join(predict_dir, 'val_predictions')

        torch.save(self.validation_step_outputs, os.path.join(predict_dir, f'{self.current_epoch}val.pkl'))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size']/self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        return [optimizer],[scheduler]
