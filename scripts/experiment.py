import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import ast
import os

from collections import defaultdict

from utils.data import preprocess, word2idx, Dictionary
from utils.analysis import sent_perplexity

# Hyperparams for now
RUN_NUM = 50    # total number of runs
TARGET_NUM = 5  # number of targets to select
TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence
SIM_RANGE = range(6)  # similarity range
INTRP_SENT_NUM = 1      # number of interrupting sentences (unrelated to each other)

# data path
data_dir = './data'
result_dir = './results'
story_path = os.path.join(data_dir, "text",
                          'On a Rainy Day - by Pat Garcia.docx')
article_path = os.path.join(
    data_dir, "text", 'Can this marriage be saved -  APA Sicence Watch.docx')

story_pool_path = os.path.join(
    data_dir, "pools",
    'diverseSim_interruptions_RainyDayStory_pool_brown_allCatges_seed_1.xlsx')

article_pool_path = os.path.join(
    data_dir, 'pools',
    'diverseSim_interruptions_APAMarriageArticle_pool_brown_allCatges_seed_1.xlsx')


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
        target_cell = pool['target_sent'][idx]
        target_sent = ast.literal_eval(target_cell)[0]
        assert (sents[idx] == target_sent)

        for sim_level in SIM_RANGE:
            start_idx = seed_num - 1 # starting point of intrp sents selection
            sim_sents = pool[f'sents_sim_intrp_{target_type}_bin{sim_level+1}_all'][idx]
            sim_sents = ast.literal_eval(sim_sents)[start_idx: start_idx + intrp_sent_num]
            sim_scores = pool[f'sim_intrp_{target_type}_bin{sim_level+1}_all'][idx]
            sim_scores = ast.literal_eval(sim_scores)[start_idx: start_idx + intrp_sent_num]

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

        if idx in target_inds:
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
        pd.Series(target_inds.tolist()), pd.Series(target_sent_all), 
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

    results.to_csv(f'./results/{intrp_sent_num}-sentence interruption/ppl_LSTM_{model_size}_results_{target_type}_seed_{seed_num}.csv', index=False)

# Run experiments

model_100_file = './data/LSTM_40m/LSTM_100_40m_a_0-d0.2.pt'
model_400_file = './data/LSTM_40m/LSTM_400_40m_a_10-d0.2.pt'
model_1600_file = './data/LSTM_40m/LSTM_1600_40m_a_20-d0.2.pt'

model_100 = torch.load(model_100_file, map_location=torch.device('cpu'))
model_100.eval()
model_400 = torch.load(model_400_file, map_location=torch.device('cpu'))
model_400.eval()

model_1600 = torch.load(model_1600_file, map_location=torch.device('cpu'))
model_1600.eval()

vocab_file = "./data/vocab.txt"
vocab = Dictionary(vocab_file)


for model, model_size in zip([model_100, model_400, model_1600],
                                [100,400,1600]):
    print(f"Model {model_size}")
    for i in tqdm(range(RUN_NUM-INTRP_SENT_NUM+1)):
        run("story", model_100, model_size, vocab, seed_num=i+1, intrp_sent_num=INTRP_SENT_NUM)
        run("article", model_100, model_size, vocab, seed_num=i+1, intrp_sent_num=INTRP_SENT_NUM)