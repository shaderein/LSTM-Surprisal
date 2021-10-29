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

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Hyperparams for now
RUN_NUM = 20    # total number of runs

INTRP_SENT_NUM = 4  # number of interrupting sentences (unrelated to each other)

TARGET_NUM = 5  # number of targets to select
TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence
TARGET_DISTANCE = 0  # target distance after the interruption. (aka offset) 
                     # TARGET_DISTANCE=0 -> the target sentence is the 1st sent
                     # right after the interruption !!!!!!!

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

    # Randomly select interruption point from the raw text.
    # In the experiment, the input text stream will be interrupted before
    # the model reads the intrp_inds-th sentence.
    # Therefore, if TARGET_DISTANCE=0, intrp_inds are exactly the target
    # sentence indices.
    np.random.seed(seed_num)
    intrp_inds = np.random.randint(TARGET_IDX_LOW, len(sents), TARGET_NUM)

    # interruption data as a dictionary
        # {(intrp_idx, sim_level) : ([intrp_sent(s)], [score(s)])}
        # where sim_level ranges from 1-10
    intrp_data = {}

    for idx in intrp_inds:
        # locate the row of the interrupted sentence, right before 
        # which the interrupting sentence will be inserted, in the pool
        if PARAPHRASE:
            intrpted_sent = pool['target_sent'][idx]
        else:
            target_cell = pool['target_sent'][idx]
            intrpted_sent = ast.literal_eval(target_cell)[0]

        assert (sents[idx] == intrpted_sent)

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

    return intrp_inds, intrp_data


def run(target_type, model, model_size, vocab, seed_num, intrp_sent_num=1):

    if target_type == 'story':
        sents_text, _ = preprocess(story_path)
    elif target_type == 'article':
        sents_text, _ = preprocess(article_path)

    _, sents = word2idx(sents_text, vocab)
    intrp_inds, intrp_data = prepare_input(target_type, seed_num, intrp_sent_num)
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

        # Append interrupting sentence into the context
        if idx + 1 in intrp_inds:
            hid_intrp_all = []  # hidden states after viewing the intrp sents
                                 # reset at each each interruption point
            for sim_level in SIM_RANGE:
                sents_intrp, scores_intrp = intrp_data[(idx + 1, sim_level)]
                # logging
                sents_intrp_all[idx+1].append(sents_intrp)
                sim_scores_all[idx+1].append(scores_intrp)

                _, sents_intrp = word2idx(sents_intrp, vocab)  # embedding
                sents_intrp = torch.cat(sents_intrp)
                # context before this point is uninterrupted
                ppl_intrp, out_intrp, hid_intrp = sent_perplexity(
                    sents_intrp, model, vocab, hid_unintrp)
                hid_intrp_all.append(hid_intrp)

        # Append the sentences into the context if it's between the 
        # interrupting sents and the actual target sentences. 
        # (i.e. nothing to append if TARGET_DISTANCE=0)
        for i in range(TARGET_DISTANCE):
            if idx in intrp_inds+i:
                for sim_level in SIM_RANGE:
                    ppl_intrp, out_intrp, hid_intrp = sent_perplexity(
                        sent, model, vocab, hid_intrp_all[sim_level])
                    # Update the context. Now the context contains:
                    #   intact context before the interruption point
                    # + interruptiong sentence
                    # + (i+1)-th sentence after the interruption
                    hid_intrp_all[sim_level] = hid_intrp

        # Log PPL's when reaching the actual target sentences
        if idx in intrp_inds+TARGET_DISTANCE:
            for sim_level in SIM_RANGE:
                ppl_intrp, out_intrp, hid_intrp = sent_perplexity(
                    sent, model, vocab, hid_intrp_all[sim_level])

                # logging
                ppl_intrp_all[idx].append(ppl_intrp.item())

            target_sent_all.append(sents_text[idx])
            ppl_base_all.append(ppl_base.item())
            ppl_unintrp_all.append(ppl_unintrp.item())

    # save data
    results = pd.concat([
        pd.Series((intrp_inds+TARGET_DISTANCE).tolist()), pd.Series(target_sent_all), 
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
            f"{intrp_sent_num}-paraphrase interruption Target {TARGET_DISTANCE+1}/")
    else:
        saved_folder = os.path.join(result_dir, 
            f"{intrp_sent_num}-sentence interruption Target {TARGET_DISTANCE+1}/")
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