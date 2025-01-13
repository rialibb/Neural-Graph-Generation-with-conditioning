import os
from tqdm import tqdm
import random
import re
import torch
from transformers import AutoTokenizer, AutoModel

random.seed(32)


def extract_numbers(text, LM):
    if not LM:
        # Use regular expression to find integers and floats
        numbers = re.findall(r'\d+\.\d+|\d+', text)
        # Convert the extracted numbers to float
        return [float(num) for num in numbers]
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        model = AutoModel.from_pretrained("bert-base-uncased")
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
        


def extract_feats(file, LM):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    #stats = extract_numbers(line, LM)
    fread.close()
    #return stats
    return line  ### ADDED