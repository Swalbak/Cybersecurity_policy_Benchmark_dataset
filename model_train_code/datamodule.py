from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import Sec_Dataset

class Sec_Data_Module(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size: int=16, max_token_len: int=512, model_name='xlm-roberta-base'):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Sec_Dataset(self.train_path, self.tokenizer, self.max_token_len)
            self.val_dataset = Sec_Dataset(self.val_path, self.tokenizer, self.max_token_len)
        if stage == 'predict':
            self.val_dataset = Sec_Dataset(self.val_path, self.tokenizer, self.max_token_len)
  
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
