# Adapted from repo "neural-perplexity"

import docx
import torch
import numpy as np
import nltk
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize


def preprocess(file_path, debug_mode=False):
    """
    Get preprocessed sentences of the full text. (Lowercase)
    return full_text: array of sentences
            is_highlight: array of binary indicators (True/False)
    """

    document = docx.Document(file_path)

    # find highlights position
    highlights = []
    full_text = []
    for paragraph in document.paragraphs:
        sentences = sent_tokenize(paragraph.text)
        full_text = full_text + sentences
        
        highlight = ""
        for run in paragraph.runs:
            if run.font.highlight_color:
                highlight += run.text
        if highlight:
            highlights = highlights + sent_tokenize(highlight)

    # FIXME: sentences makred by their contents regardless of positions
    is_hightlight = [True if s in highlights else False for s in full_text]

    if debug_mode:
        for i in range(len(full_text)):
            print(f"{is_hightlight[i]} {full_text[i]}")

    return full_text, is_hightlight

def word2idx(sents, vocab):
    """
    tokenize sentences into list of word idx
    """
    all_ids = []
    processed_text = []
    for s in sents:
        # Note: should add eos to reduce PPL
        # Note: all sentences start with <eos>
        words = ['<eos>'] + word_tokenize(s.lower()) 
        ids = torch.LongTensor(len(words))
        processed_words = words
        for i, word in enumerate(words):
            # unk words that are OOV
            if word not in vocab.word2idx:
                ids[i] = vocab.word2idx['<unk>']
                processed_words[i] ='<unk>'
            else:
                ids[i] = vocab.word2idx[word]
        processed_text.append(processed_words)
        all_ids.append(ids.unsqueeze(dim=1))

    # all_ids: a list of tensors with shape: (sent_len, batch_size=1)
    return processed_text, all_ids

class Dictionary(object):
    def __init__(self, path=None):
        self.word2idx = {}
        self.idx2word = []
        if path:
            self.load(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.add_word(line.rstrip('\n'))

    def save(self, path):
        with open(path, 'w') as f:
            for w in self.idx2word:
                f.write('{}\n'.format(w))