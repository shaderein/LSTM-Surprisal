from nltk.corpus.reader.chasen import test
import numpy as np
import torch
import torch.nn as nn
from utils.data import Dictionary
from torch.distributions import Categorical

def sent_measurements(sent, model, vocab: Dictionary, hidden_init):
    """
    Params:
        sent: a tensor of words index; shape = <sent_len=n, batch_size=1>
              The first token (w_0) is always <eos>
    Return:
        ppl: the perplexity of the entire sentence (w_1, ... w_n-1)
    """
    vocab_size = len(vocab.idx2word)

    target = sent.squeeze(dim=1)[1:]  # convert to 1D array

    out, hidden = model(sent, hidden_init)
    out = out.squeeze(dim=1)

    pred = out[:-1]

    # Perplexity (surprisal)
    ce = nn.functional.cross_entropy(pred, target)
    ppl = torch.exp(ce)

    # Entropy (confidence)
    softmax_layer = nn.Softmax(dim=1)
    normalized_pred = softmax_layer(pred)
    entropy = Categorical(probs=normalized_pred).entropy().mean()

    return ppl, out, hidden, entropy
