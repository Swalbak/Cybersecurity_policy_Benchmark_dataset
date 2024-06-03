import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class Sec_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_token_len :int=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        self._prepare_data()
    
    def _prepare_data(self):
        if type(self.data_path) != str:
            self.data = self.data_path
        else:
            self.data = pd.read_pickle(self.data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        abstract = item['chunked_text']
        labels = item['label']
        tokens = self.tokenizer.encode_plus(abstract,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=self.max_token_len,
                                            padding='max_length',
                                            return_attention_mask=True
                                            )
        
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}
