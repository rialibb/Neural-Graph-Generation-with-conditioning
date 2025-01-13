import os
from tqdm import tqdm
import random
import re
import torch
from transformers import AutoTokenizer, AutoModel

random.seed(32)


def extract_numbers(text):

    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]

        


def extract_feats(file):

    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    fread.close()
    return line  