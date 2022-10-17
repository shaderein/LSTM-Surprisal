from functools import reduce
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import os
import pickle

from utils.data import preprocess, word2idx, Dictionary
from utils.analysis import sent_measurements

# Load model

model_file = './data/LSTM_40m/LSTM_1600_40m_a_20-d0.2.pt'

model = torch.load(model_file, map_location=torch.device('cpu'))
model.eval()

# Load vocab and text
data_dir = './data'
result_dir = './results'

article_path = os.path.join(
    data_dir, "text", 'Can this marriage be saved -  APA Sicence Watch.docx')

vocab_file = "./data/vocab.txt"
vocab = Dictionary(vocab_file)

sents_text, _ = preprocess(article_path)
_, sents = word2idx(sents_text, vocab)
hid_size = model.nhid
hid_init = model.init_hidden(bsz=1)

# logging
target_sent_all = []
ppl_base_all = []
ppl_unintrp_all = []

for idx in range(len(sents)):
    sent = sents[idx]

    if idx == 0:
        out, hid = model(sent, hid_init)
        hid_unintrp, hid_intrp = hid, hid
        continue

    # Base PPL
    ppl_base, out_base, hid_base = sent_measurements(sent, model, vocab,
                                                    hid_init)

    # Uninterrupted PPL (with context)
    ppl_unintrp, out_unintrp, hid_unintrp = sent_measurements(
        sent, model, vocab, hid_unintrp)

    target_sent_all.append(sents_text[idx])
    ppl_base_all.append(ppl_base.item())
    ppl_unintrp_all.append(ppl_unintrp.item())

# save data
results = pd.concat([
    pd.Series(target_sent_all), 
    pd.Series(ppl_base_all), 
    pd.Series(ppl_unintrp_all),  
    ],
    keys=[
        "target_sent", 
        "PPL_base",
        "PPL_w_context",
    ],
    axis=1)

results.to_csv(f'./results/ppl_article_for_EB_LSTM_1600.csv', index=False)
