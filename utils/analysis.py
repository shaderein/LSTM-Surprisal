import numpy as np
import torch
import torch.nn as nn
from utils.data import Dictionary
import model

def sent_perplexity(sent, model, vocab:Dictionary, out_prev, hidden_init):
    """
    Params:
        sent: a tensor of words index; shape = <batch_size=1, sent_len>
        out_prev: output from the network at the last timestep of the previous sentence
    Return:
        ppl: the perplexity of the entire sentence (w_0, ... w_n-1)
    """
    vocab_size = len(vocab.idx2word)
    
    target = sent.squeeze(dim=1) # convert to 1D array
    
    out, hidden = model(sent, hidden_init)
    out = out.squeeze(dim=1)

    pred = torch.zeros(len(sent), vocab_size)
    pred[0] = out_prev # include the last output from the previous sentence
                        # which predicts the first word of the current sent 
    pred[1:] = out[:-1]

    ce = nn.functional.cross_entropy(pred, target)
    ppl = torch.exp(ce)
    return ppl, out, hidden

        

