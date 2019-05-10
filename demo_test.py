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
max_seq_length = 128
text_a = "what a stupid person, he is from america."
target = "stupid"
token_target = tokenizer.tokenize(target)
token_a = tokenizer.tokenize(text_a)
tokens = ["[CLS]"] + token_a + ["[SEP]"]
label_id = 1
segment_ids = [label_id] * len(tokens)
masked_lm_labels = [-1]*max_seq_length

output_tokens = list(tokens)
masked_index = tokens.index(target)
masked_lm_labels[masked_index] = tokenizer.convert_tokens_to_ids([tokens[masked_index]])[0]
output_tokens[masked_index] = '[MASK]'
# Mask a token that we will try to predict back with `BertForMaskedLM`

init_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
padding = [0] * (max_seq_length - len(input_ids))
init_ids += padding
input_ids += padding
input_mask += padding
segment_ids += padding

assert len(init_ids) == max_seq_length
assert len(input_ids) == max_seq_length
assert len(input_mask) == max_seq_length
assert len(segment_ids) == max_seq_length

all_init_ids = torch.tensor([init_ids], dtype=torch.long)
all_input_ids = torch.tensor([input_ids], dtype=torch.long)
all_input_mask = torch.tensor([input_mask], dtype=torch.long)
all_segment_ids = torch.tensor([segment_ids], dtype=torch.long)


MODEL_name = "{}/BertForMaskedLM_aug{}_epoch_3".format('toxic', 'toxic')
#model = load_model(MODEL_name)
model = BertForMaskedLM.from_pretrained(modelpath)
#model.cuda()
model.eval()

predictions = model(all_init_ids, all_segment_ids, all_input_mask)

print(predictions)
pred = torch.argsort(predictions)[:,-1]
#id = pred
str = tokenizer.convert_ids_to_tokens(27580)
print(str)