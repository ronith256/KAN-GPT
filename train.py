import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from mingpt.utils import set_seed
set_seed(42)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.kan import *

import math
from mingpt.utils import sample
import argparse
from torch.utils.data import Dataset


from mingpt.model import GPT, GPTConfig
from mingpt.kan import *

from mingpt.trainer import Trainer, TrainerConfig


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size:d} characters, {vocab_size:d} unique.")

        self.stoi = { ch: i for i, ch in enumerate(chars) }
        self.itos = { i: ch for i, ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx+self.block_size+1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
       
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GPT model')
    parser.add_argument('data_path', type=str, help='Path to data.txt')
    parser.add_argument('--load', type=str, help='Path to load pre-trained model')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--max-epochs', type=int, default=2, help='Maximum number of epochs')
    parser.add_argument('--output', type=str, default='kan-gpt.pth', help='Path to save model')
    parser.add_argument('--generate', type=bool, default=True, help='Generate sample output from model')

    args = parser.parse_args()

    data_path = args.data_path
    load_path = args.load
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    output = args.output

    text = open(data_path, 'r').read()
    train_dataset = CharDataset(text, block_size=128)

    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=4,
        n_head=8,
        n_embd=512,
    )
    model = GPT(mconf)

    block_size = 128

    if load_path:
        model.load_state_dict(torch.load(load_path))

    tconf = TrainerConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=2*len(train_dataset)*block_size,
        num_workers=4,
        ckpt_path=output, 
        generate = args.generate
    )
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()
