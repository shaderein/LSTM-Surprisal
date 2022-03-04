import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
import ast
import os

from collections import defaultdict, OrderedDict

from utils.data import preprocess, word2idx, Dictionary
from utils.analysis import sent_perplexity
from utils.load import create_folders_if_necessary, load_text_path, load_models

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Hyperparams for now

TARGET_NUM = 5  # number of targets to select

# Temporarily disable this since only a few sentences are selected
#   at the begining of the text
# TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence

SIM_RANGE = range(6)  # similarity range
PARAPHRASE = False   # Use paraphrase sentences for interruption

INTRP_SENT_NUM = 4  # number of interrupting sentences (unrelated to each other)
TARGET_DISTANCE = 4  # target distance after the interruption. (aka offset) 
                     # TARGET_DISTANCE=0 -> the target sentence is the 1st sent
                     # right after the interruption !!!!!!!

RUN_NUM = 47    # total number of runs

conditions = [
              (1, 0, 48),
              (2, 0, 23),
              (4, 0, 11),
              (4, 1, 11),
              (4, 2, 11)
            ]

for INTRP_SENT_NUM, TARGET_DISTANCE, RUN_NUM in conditions:

    print(f"{INTRP_SENT_NUM}-sent Target{TARGET_DISTANCE}")

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
        intrp_inds = np.random.randint(1, len(sents)-2, TARGET_NUM)

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
                    sim_sents_header = \
                        f'sents_sim_intrp_{target_type}_bin{sim_level+1}_all'
                    sim_scores_header = \
                        f'sim_intrp_{target_type}_bin{sim_level+1}_all'

                # intrp sents sampling
                start_idx = seed_num * intrp_sent_num + 1 
                end_idx = (seed_num+1) * intrp_sent_num + 1 

                sim_sents = pool[sim_sents_header][idx]
                sim_sents = ast.literal_eval(sim_sents)\
                                [start_idx: end_idx]
                sim_scores = pool[sim_scores_header][idx]
                if PARAPHRASE:
                    sim_scores = re.sub(" +", ",", sim_scores)
                    sim_scores = sim_scores.replace(",]", "]")
                sim_scores = ast.literal_eval(sim_scores)\
                                [start_idx: end_idx]

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
        target_sent_all = defaultdict(list)
        # key = target_idx, value = value in all bins
        sents_intrp_all = defaultdict(list)
        sim_scores_all = defaultdict(list)
        ppl_base_all = defaultdict(list)
        ppl_loc_unintrp_all = defaultdict(list)
        ppl_loc_intrp_all = defaultdict(list) # PPL at the target sentence after 
                                        # viewing only the interrupting 
                                        # sentences (local context)
        ppl_glo_unintrp_all = defaultdict(list)
        ppl_glo_intrp_all = defaultdict(list)

        for idx in range(len(sents)):
            sent = sents[idx]

            if idx == 0:
                out, hid = model(sent, hid_init)
                continue

            # hid contains the cummulative context of the intact text
            _, _, hid = sent_perplexity(sent, model, vocab, hid)

            if idx + 1 not in intrp_inds:
                continue

            # ---
            # Reach the interruption point
            # ---
            hid_glo_intrp_all = []  # hidden states after viewing the intrp sents
                                    # and the global context

            hid_loc_intrp_all = []  # hidden states after viewing only the intrp
                                #   sents without any preceding global context

            hid_loc_unintrp = hid_init

            hid_glo_unintrp = hid

            ## Feed in Interrupting Sentences
            for sim_level in SIM_RANGE:
                sents_intrp, scores_intrp = intrp_data[(idx + 1, sim_level)]
                # logging
                sents_intrp_all[idx+1].append(sents_intrp)
                sim_scores_all[idx+1].append(scores_intrp)

                _, sents_intrp = word2idx(sents_intrp, vocab)  # embedding
                sents_intrp = torch.cat(sents_intrp)

                # contains S_global, S_intrp
                _, _, hid_glo_intrp = sent_perplexity(
                    sents_intrp, model, vocab, hid_glo_unintrp)
                hid_glo_intrp_all.append(hid_glo_intrp)

                # contains S_intrp only
                _, _, hid_loc_intrp = sent_perplexity(
                    sents_intrp, model, vocab, hid_init)
                hid_loc_intrp_all.append(hid_loc_intrp)

            ## Feed in Local Context
            ## (Skip if TARGET_DISTANCE=0 : targets are immediate)
            if TARGET_DISTANCE > 0:
                sents_loc = torch.cat(sents[idx+1 : idx+1+TARGET_DISTANCE])

                # contains S_global, S_local
                _, _, hid_glo_unintrp = sent_perplexity(
                    sents_loc, model, vocab, hid_glo_unintrp)

                # contains S_local only
                _, _, hid_loc_unintrp = sent_perplexity(
                    sents_loc, model, vocab, hid_init)

                for sim_level in SIM_RANGE:
                    # contains S_global, S_intrp, S_local
                    _, _, hid_glo_intrp = sent_perplexity(
                        sents_loc, model, vocab, hid_glo_intrp_all[sim_level])
                    hid_glo_intrp_all[sim_level] = hid_glo_intrp

                    # contains S_intrp, S_local
                    _, _, hid_loc_intrp = sent_perplexity(
                        sents_loc, model, vocab, hid_loc_intrp_all[sim_level])
                    hid_loc_intrp_all[sim_level] = hid_loc_intrp

            ## Feed in Target and log PPL
            target_idx = idx + 1 + TARGET_DISTANCE
            sent_target = sents[target_idx]
            target_sent_all[target_idx] = sents_text[target_idx]

            # PPL ( S_target )
            ppl_base, _, _ = sent_perplexity(sent_target, model, vocab, hid_init)
            ppl_base_all[target_idx] = ppl_base.item()

            # PPL ( S_target | S_global, S_local )
            ppl_glo_unintrp, _, hid_glo_unintrp = sent_perplexity(
                sent_target, model, vocab, hid_glo_unintrp)
            ppl_glo_unintrp_all[target_idx] = ppl_glo_unintrp.item()

            # PPL ( S_target | S_local )
            ppl_loc_unintrp, _, hid_loc_unintrp = sent_perplexity(
                sent_target, model, vocab, hid_loc_unintrp)
            ppl_loc_unintrp_all[target_idx] = ppl_loc_unintrp.item()

            for sim_level in SIM_RANGE:
                # PPL ( S_target | S_global, S_intrp, S_local )
                ppl_glo_intrp, _, hid_glo_intrp = sent_perplexity(
                    sent_target, model, vocab, hid_glo_intrp_all[sim_level])
                ppl_glo_intrp_all[idx].append(ppl_glo_intrp.item())

                # # PPL ( S_target | S_intrp, S_local )
                ppl_local_intrp, _, hid_loc_intrp = sent_perplexity(
                    sent_target, model, vocab, hid_loc_intrp_all[sim_level])
                ppl_loc_intrp_all[idx].append(ppl_local_intrp.item())

        # save data
        # Notes: all dicts are sorted since items are inserted as the original
        #        text is traversed. Need to explicitly sort the dict if the
        #        implementation changes later
        results = pd.concat([
            pd.Series(target_sent_all.keys()), 
            pd.Series(target_sent_all.values()), 
            pd.Series(sents_intrp_all.values()),
            pd.Series(sim_scores_all.values()), 
            pd.Series(ppl_base_all.values()), 
            pd.Series(ppl_loc_unintrp_all.values()),  
            pd.Series(ppl_loc_intrp_all.values()),
            pd.Series(ppl_glo_unintrp_all.values()),  
            pd.Series(ppl_glo_intrp_all.values())
        ],
        keys=[
            "target_idx", 
            "target_sent", 
            "sents_intrp_all_bins",
            "sim_intrp_all_bins",
            "base_PPL",
            "local_unintrp_PPL",
            "local_intrp_PPL_all_bins",
            "global_unintrp_PPL",
            "global_intrp_PPL_all_bins"
        ],
        axis=1)

        if PARAPHRASE: 
            saved_folder = os.path.join(result_dir, 
                f"{intrp_sent_num}-paraphrase_Target-{TARGET_DISTANCE+1}/")
        else:
            saved_folder = os.path.join(result_dir, 
                f"{intrp_sent_num}-sentence_Target-{TARGET_DISTANCE+1}/")
        saved_file_name = \
            f'ppl_LSTM_{model_size}_{target_type}_seed_{seed_num}.csv'
        create_folders_if_necessary(saved_folder)

        results.to_csv(os.path.join(saved_folder, saved_file_name), index=False)

    # Run experiments
    for model in models:
        model_size = model.nhid
        print(f"Model {model_size}")
        for i in tqdm(range(RUN_NUM)):
            run("article", model, model_size, vocab, seed_num=i+1, 
                intrp_sent_num=INTRP_SENT_NUM)
            run("story", model, model_size, vocab, seed_num=i+1, 
                intrp_sent_num=INTRP_SENT_NUM)