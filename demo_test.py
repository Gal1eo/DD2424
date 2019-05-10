"Fine-tuning BertMasked Model with labeled dataset"
from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from tqdm import trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

logger = logging.getLogger(__name__)


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

def load_model(model_name):
    weights_path = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, model_name)
    model = torch.load(weights_path)
    return model

# Load pre-trained model tokenizer (vocabulary)
modelpath = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(modelpath)

text = "dummy. he is a racist, he hates black people."
target = "black"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = tokenized_text.index(target)
tokenized_text[masked_index] = '[MASK]'

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [1] * len(tokenized_text)
# this is for the dummy first sentence.
segments_ids[0] = 0
segments_ids[1] = 0

# Convert inputs to PyTorch tensors


tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')



MODEL_name = "{}/BertForMaskedLM_{}_epoch_10".format('toxic', 'toxic')
model = load_model(MODEL_name)
model.cuda()

model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])

print("Original:", text)
print("Masked:", " ".join(tokenized_text))

print("Predicted token:", predicted_token)
print("Other options:")
# just curious about what the next few options look like.
for i in range(10):
    predictions[0,masked_index,predicted_index] = -11100000
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    print(predicted_token)

