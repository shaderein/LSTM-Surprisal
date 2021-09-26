import pandas as pd
import numpy as np
import torch
import ast
import os

from collections import defaultdict

from utils.data import preprocess, word2idx, Dictionary
from utils.analysis import sent_perplexity

# Hyperparams for now
TARGET_NUM = 5  # number of targets to select
TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence
SIM_RANGE = range(3)  # similarity range (currently 0(low)-2(high))

# data path
data_dir = './data'
result_dir = './results'
story_path = os.path.join(data_dir, "text",
                          'On a Rainy Day - by Pat Garcia.docx')
article_path = os.path.join(
    data_dir, "text", 'Can this marriage be saved -  APA Sicence Watch.docx')

story_pool_path = os.path.join(
    data_dir, "pools",
    'interruption_sim_RainyDayStory_pool_brown_allCatges_seed_1.xlsx')

article_pool_path = os.path.join(
    data_dir, 'pools',
    'interruption_sim_APAMarriageArticle_pool_brown_allCatges_seed_1.xlsx')


def prepare_input(target_type, seed_num):
    if target_type == "story":
        sents, _ = preprocess(story_path)
        pool = pd.read_excel(story_pool_path)
    elif target_type == "article":
        sents, _ = preprocess(article_path)
        pool = pd.read_excel(article_pool_path)

    # randomly select target sents from the raw text
    np.random.seed(seed_num)
    target_inds = np.random.randint(TARGET_IDX_LOW, len(sents), TARGET_NUM)
    intrp_data = {}

    for idx in target_inds:
        target_cell = pool['target_sent'][idx]
        target_sent = ast.literal_eval(target_cell)[0]
        assert (sents[idx] == target_sent)

        low_sim_sent = pool['intrp_sentence_low_sim'][idx]
        # QS: 0-based or 1-based?
        low_sim_sent = ast.literal_eval(low_sim_sent)[seed_num - 1]
        low_sim_score = pool['sim_score_low'][idx]
        low_sim_score = ast.literal_eval(low_sim_score)[seed_num - 1]

        mid_sim_sent = pool['intrp_sentence_mid_sim'][idx]
        mid_sim_sent = ast.literal_eval(mid_sim_sent)[seed_num - 1]
        mid_sim_score = pool['sim_score_mid'][idx]
        mid_sim_score = ast.literal_eval(mid_sim_score)[seed_num - 1]

        high_sim_sent = pool['intrp_sentence_high_sim'][idx]
        high_sim_sent = ast.literal_eval(high_sim_sent)[seed_num - 1]
        high_sim_score = pool['sim_score_high'][idx]
        high_sim_score = ast.literal_eval(high_sim_score)[seed_num - 1]

        # interruption data as a dictionary
        # {(target_idx, sim_level) : (intrp_sent, score)}
        # where sim_level ranges from 1-10

        intrp_data[(idx, 0)] = (low_sim_sent, low_sim_score)
        intrp_data[(idx, 1)] = (mid_sim_sent, mid_sim_score)
        intrp_data[(idx, 2)] = (high_sim_sent, high_sim_score)

    return target_inds, intrp_data


def run(target_type, model, vocab, seed_num):

    if target_type == 'story':
        sents_text, _ = preprocess(story_path)
    elif target_type == 'article':
        sents_text, _ = preprocess(article_path)

    _, sents = word2idx(sents_text, vocab)
    target_inds, intrp_data = prepare_input(target_type, seed_num)
    hid_size = model.nhid
    hid_init = model.init_hidden(bsz=1)

    # logging
    target_sent_all = []
    sent_intrp_all = defaultdict(list)
    sim_score_all = defaultdict(list)
    ppl_base_all = []
    ppl_unintrp_all = []
    ppl_intrp_all = defaultdict(list)

    for idx in range(len(sents)):
        sent = sents[idx]

        if idx == 0:
            out, hid = model(sent, hid_init)
            hid_unintrp, hid_intrp = hid, hid
            continue

        # Base PPL
        ppl_base, out_base, hid_base = sent_perplexity(sent, model, vocab,
                                                       hid_init)

        # Uninterrupted PPL (with context)
        ppl_unintrp, out_unintrp, hid_unintrp = sent_perplexity(
            sent, model, vocab, hid_unintrp)

        # Interupted PPL (with context)

        # ordered by the similarity level (low->high)
        if idx + 1 in target_inds:
            hid_intrp_prev = []  # hidden states after viewing the intrp sents
            for sim_level in SIM_RANGE:
                sent_intrp, score_intrp = intrp_data[(idx + 1, sim_level)]
                # logging
                sent_intrp_all[sim_level].append(sent_intrp)
                sim_score_all[sim_level].append(score_intrp)

                _, sent_intrp = word2idx([sent_intrp], vocab)  # embedding
                sent_intrp = sent_intrp[0]
                ppl_intrp, out_intrp, hid_intrp = sent_perplexity(
                    sent_intrp, model, vocab, hid_unintrp)
                hid_intrp_prev.append(hid_intrp)

        if idx in target_inds:
            for sim_level in SIM_RANGE:
                ppl_intrp, out_intrp, hid_intrp = sent_perplexity(
                    sent, model, vocab, hid_intrp_prev[sim_level])

                # logging
                ppl_intrp_all[sim_level].append(ppl_intrp.item())

            target_sent_all.append(sents_text[idx])
            ppl_base_all.append(ppl_base.item())
            ppl_unintrp_all.append(ppl_unintrp.item())

    # save data
    results = pd.concat([
        pd.Series(target_inds.tolist()), pd.Series(target_sent_all), 
        pd.Series(sent_intrp_all[0]), 
        pd.Series(sent_intrp_all[1]),
        pd.Series(sent_intrp_all[2]), 
        pd.Series(sim_score_all[0]), 
        pd.Series(sim_score_all[1]), 
        pd.Series(sim_score_all[2]),
        pd.Series(ppl_base_all), 
        pd.Series(ppl_unintrp_all),  
        pd.Series(ppl_intrp_all[0]), 
        pd.Series(ppl_intrp_all[1]),
        pd.Series(ppl_intrp_all[2])
    ],
    keys=[
        "target index", "target sentence", 
        "low sim interruption sentene", 
        "mid sim interruption sentene", 
        "high sim interruption sentene",
        "low sim score", 
        "mid sim score", 
        "high sim score",
        "base PPL",
        "uninterrupted PPL",
        "interrupted PPL low sim",
        "interrupted PPL mid sim",
        "interrupted PPL high sim",
    ],
    axis=1)

    results.to_csv(f'./results/ppl_{target_type}_seed_{seed_num}.csv', index=False)



model_100_file = './data/LSTM_40m/LSTM_100_40m_a_0-d0.2.pt'
model_100 = torch.load(model_100_file, map_location=torch.device('cpu'))
model_100.eval()

vocab_file = "./data/vocab.txt"
vocab = Dictionary(vocab_file)

run("story", model_100, vocab, seed_num=1)
run("article", model_100, vocab, seed_num=1)