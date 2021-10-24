import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
import ast
import os

from collections import defaultdict

from utils.data import preprocess, word2idx, Dictionary
from utils.analysis import sent_perplexity
from utils.load import create_folders_if_necessary, load_text_path, load_models

# Hyperparams for now
RUN_NUM = 20    # total number of runs
INTRP_SENT_NUM = 4  # number of interrupting sentences (unrelated to each other)
TARGET_NUM = 5  # number of targets to select
TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence
TARGET_DISTANCE = 5  # target distance after the interruption. 1=1st sent right after it.
SIM_RANGE = range(6)  # similarity range
PARAPHRASE = False   # Use paraphrase sentences for interruption

# Modified above for different experiment conditions
assert(RUN_NUM+INTRP_SENT_NUM-1 <= 50)
if PARAPHRASE:
    assert(SIM_RANGE==range(1))

# model size of interested. Should be in [100,400,1600]
models_size = [1600]

# saving path
result_dir = './results'

# load models
models = load_models(models_size)

# load text path and vocab
story_path, article_path, \
story_pool_path, article_pool_path = load_text_path(PARAPHRASE)

vocab_file = "./data/vocab.txt"
vocab = Dictionary(vocab_file)


def prepare_input(target_type, seed_num, intrp_sent_num):
    if target_type == "story":
        sents, _ = preprocess(story_path)
        pool = pd.read_excel(story_pool_path)
    elif target_type == "article":
        sents, _ = preprocess(article_path)
        pool = pd.read_excel(article_pool_path)

    # randomly select target sents from the raw text
    np.random.seed(seed_num)
    target_inds = np.random.randint(TARGET_IDX_LOW, len(sents), TARGET_NUM)

    # interruption data as a dictionary
        # {(target_idx, sim_level) : ([intrp_sent(s)], [score(s)])}
        # where sim_level ranges from 1-10
    intrp_data = {}

    for idx in target_inds:
        # get the target sentence
        if PARAPHRASE:
            target_sent = pool['target_sent'][idx]
        else:
            target_cell = pool['target_sent'][idx]
            target_sent = ast.literal_eval(target_cell)[0]

        assert (sents[idx] == target_sent)

        # get interrupting sentences
        for sim_level in SIM_RANGE:
            if PARAPHRASE:
                sim_sents_header = 'paraphrased'
                sim_scores_header = 'sim_scores'
            else:
                sim_sents_header = f'sents_sim_intrp_{target_type}_bin{sim_level+1}_all'
                sim_scores_header = f'sim_intrp_{target_type}_bin{sim_level+1}_all'
            start_idx = seed_num - 1 # starting point of intrp sents selection
            sim_sents = pool[sim_sents_header][idx]
            sim_sents = ast.literal_eval(sim_sents)\
                            [start_idx: start_idx + intrp_sent_num]
            sim_scores = pool[sim_scores_header][idx]
            if PARAPHRASE:
                sim_scores = re.sub(" +", ",", sim_scores)
                sim_scores = sim_scores.replace(",]", "]")
            sim_scores = ast.literal_eval(sim_scores)\
                            [start_idx: start_idx + intrp_sent_num]

            intrp_data[(idx, sim_level)] = (sim_sents, sim_scores)

    return target_inds, intrp_data


def run(target_type, model, model_size, vocab, seed_num, intrp_sent_num=1):

    if target_type == 'story':
        sents_text, _ = preprocess(story_path)
    elif target_type == 'article':
        sents_text, _ = preprocess(article_path)

    _, sents = word2idx(sents_text, vocab)
    target_inds, intrp_data = prepare_input(target_type, seed_num, intrp_sent_num)
    hid_size = model.nhid
    hid_init = model.init_hidden(bsz=1)

    # logging
    target_sent_all = []
    # key = target_idx, value = value in all bins
    sents_intrp_all = defaultdict(list)
    sim_scores_all = defaultdict(list)
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
                sents_intrp, scores_intrp = intrp_data[(idx + 1, sim_level)]
                # logging
                sents_intrp_all[idx+1].append(sents_intrp)
                sim_scores_all[idx+1].append(scores_intrp)

                _, sents_intrp = word2idx(sents_intrp, vocab)  # embedding
                sents_intrp = torch.cat(sents_intrp)
                ppl_intrp, out_intrp, hid_intrp = sent_perplexity(
                    sents_intrp, model, vocab, hid_unintrp)
                hid_intrp_prev.append(hid_intrp)

        if idx in target_inds+(TARGET_DISTANCE-1):
            for sim_level in SIM_RANGE:
                ppl_intrp, out_intrp, hid_intrp = sent_perplexity(
                    sent, model, vocab, hid_intrp_prev[sim_level])

                # logging
                ppl_intrp_all[idx].append(ppl_intrp.item())

            target_sent_all.append(sents_text[idx])
            ppl_base_all.append(ppl_base.item())
            ppl_unintrp_all.append(ppl_unintrp.item())

    # save data
    results = pd.concat([
        pd.Series((target_inds+(TARGET_DISTANCE-1)).tolist()), pd.Series(target_sent_all), 
        pd.Series(sents_intrp_all.values()),
        pd.Series(sim_scores_all.values()), 
        pd.Series(ppl_base_all), 
        pd.Series(ppl_unintrp_all),  
        pd.Series(ppl_intrp_all.values())
    ],
    keys=[
        "target_idx", 
        "target_sent", 
        "sents_intrp_all_bins",
        "sim_intrp_all_bins",
        "base_PPL",
        "unintrp_PPL",
        "intrp_PPL_all_bins"
    ],
    axis=1)

    if PARAPHRASE: 
        saved_folder = os.path.join(result_dir, 
                                    f"{intrp_sent_num}-paraphrase interruption Target {TARGET_DISTANCE}/")
    else:
        saved_folder = os.path.join(result_dir, 
                                    f"{intrp_sent_num}-sentence interruption Target {TARGET_DISTANCE}/")
    saved_file_name = f'ppl_LSTM_{model_size}_{target_type}_seed_{seed_num}.csv'
    create_folders_if_necessary(saved_folder)

    results.to_csv(os.path.join(saved_folder, saved_file_name), index=False)

# Run experiments
for model in models:
    model_size = model.nhid
    print(f"Model {model_size}")
    for i in tqdm(range(RUN_NUM)):
        run("story", model, model_size, vocab, seed_num=i+1, intrp_sent_num=INTRP_SENT_NUM)
        run("article", model, model_size, vocab, seed_num=i+1, intrp_sent_num=INTRP_SENT_NUM)