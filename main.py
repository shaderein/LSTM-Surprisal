from functools import reduce
import torch
import numpy as np
import torch.nn as nn
import pickle

from utils.data import preprocess, word2idx, Dictionary
from utils.analysis import sent_perplexity

# Load model

model_file = './data/LSTM_40m/LSTM_100_40m_a_0-d0.2.pt'

model = torch.load(model_file, map_location=torch.device('cpu'))
model.eval()

# Load vocab and text

data_file = "./data/Full Story_So much water so close to home_Highlighted.docx"
vocab_file = "./data/vocab.txt"

vocab = Dictionary(vocab_file)
full_text, is_highlight = preprocess(data_file)
processed_text, all_ids = word2idx(full_text, vocab)

# calculate PPL of highlighed sentences

ppl_sent_highlight = []

with torch.no_grad():
    for i, sent in enumerate(all_ids[0:100]):
        if i == 0:
            hidden = model.init_hidden(bsz=1)
            out, hidden = model(sent, hidden)
            continue
        out_prev = out[-1, 0]  #TODO: optimize memory
        ppl, out, hidden = sent_perplexity(sent, model, vocab, out_prev, hidden)
        if is_highlight[i]:
            ppl_sent_highlight.append(ppl)

    print(f"ppl per sent {np.mean(ppl_sent_highlight)}")
