import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
import ast
import os

from collections import defaultdict, OrderedDict

from utils.data import preprocess, word2idx, Dictionary
from utils.analysis import sent_measurements, generate_text
from utils.load import create_folders_if_necessary, load_text_path, load_models

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Hyperparams for now

# TARGET_NUM = 5  # number of targets to select

# Temporarily disable this since only a few sentences are selected
#   at the begining of the text
# TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence

# TODO
SIM_RANGE = range(10)  # similarity range
PARAPHRASE = False   # Use paraphrase sentences for interruption

GENERATE_LENGTH = 10 # length of text to generated after intrp

""""
INTRP_SENT_NUM:     # number of interrupting sentences (unrelated to each other)
TARGET_DISTANCE:    # target distance after the interruption. (aka offset) 
                    # TARGET_DISTANCE=0 -> the target sentence is the 1st sent
                    # right after the interruption !!!!!!!

RUN_NUM =           # total number of runs
"""

# TODO
# Batch runs: INTRP_SENT_NUM, TARGET_DISTANCE
conditions = [#(1, 0),
              (1, 1),
              (1, 2),
              (1, 3),
              (1, 4),
              (1, 5),
              (1, 6),
              (1, 7)
            ]

# TODO: can optimize the code by merge conditions where target_distance varies but intrp_sent_num is the same
for INTRP_SENT_NUM, TARGET_DISTANCE in conditions:

    print(f"{INTRP_SENT_NUM}-sent Target{TARGET_DISTANCE}")

    # model size of interested. Should be in [100,400,1600]
    models_size = [1600]

    # load models
    models = load_models(models_size)

    # load text path and vocab
    story_path, article_path, \
    story_pool_path, article_pool_path = load_text_path(PARAPHRASE) # TODO

    vocab_file = "./data/vocab.txt"
    vocab = Dictionary(vocab_file)

    def prepare_input(target_type, intrp_sent_num):
        if target_type == "story":
            sents, _ = preprocess(story_path)
            pool = pd.read_excel(story_pool_path)
        elif target_type == "article":
            sents, _ = preprocess(article_path)
            pool = pd.read_excel(article_pool_path)

        # In the experiment, the input text stream will be interrupted before
        # the model reads the intrp_inds-th sentence.
        # Therefore, if TARGET_DISTANCE=0, intrp_inds are exactly the target
        # sentence indices.
        intrp_inds = np.asarray([17, 37, 54, 66, 76, 88, 106, 118, 136, 148, 158, 171, 188, 200, 224, 232, 246]) + 1

        # interruption data as a dictionary
            # {(intrp_idx, sim_level) : ([intrp_sent(s)], [score(s)])}
            # where sim_level ranges from 0-9 # TODO
            # 0-4: low; 5-9: high
        intrp_data = {}

        for i in range(len(intrp_inds)):
            # locate the row of the interrupted sentence, right before # TODO
            # which the interrupting sentence will be inserted, in the pool

            # get interrupting sentences
            for level in range(5):

                row = i * 5 + level

                preceding_sent = pool['Primary Story (preceding sentences)'][i]

                sim_sents_low_header = \
                    f'Min Similarity ~ - 0.15'
                sim_scores_low_header = \
                    f'low sim scores'

                sim_sents_high_header = \
                    f'Max Similarity ~ 0.4 <'
                sim_scores_high_header = \
                    f'high sim scores'

                sim_sents_low = pool[sim_sents_low_header][row]
                sim_scores_low = pool[sim_scores_low_header][row]

                sim_sents_high = pool[sim_sents_high_header][row]
                sim_scores_high = pool[sim_scores_high_header][row]

                intrp_data[(intrp_inds[i], level)] = (sim_sents_low, sim_scores_low)
                intrp_data[(intrp_inds[i], level+5)] = (sim_sents_high, sim_scores_high)

        return intrp_inds, intrp_data


    def run(target_type, model, model_size, vocab, intrp_sent_num=1):

        if target_type == 'story':
            sents_text, _ = preprocess(story_path)
        elif target_type == 'article':
            sents_text, _ = preprocess(article_path)

        _, sents = word2idx(sents_text, vocab)
        intrp_inds, intrp_data = prepare_input(target_type, intrp_sent_num)
        hid_size = model.nhid
        hid_init = model.init_hidden(bsz=1)

        # logging
        target_sent_all = defaultdict(list)
        # key = intrp_idx, value = [value in all bins]
        sents_intrp_all = defaultdict(list)
        preceding_all = defaultdict(list) # Note: new
        sim_scores_all = defaultdict(list)

        # collect generated text after intrp
        generated_unintrp_all = defaultdict(list)
        generated_intrp_all = defaultdict(list)

        # key = intrp_idx, 
        # value = {target_distance : [value in all bins]}
        ppl_base_all = defaultdict(list)
        ppl_loc_unintrp_all = defaultdict(list)
        ppl_loc_intrp_all = defaultdict(list) # PPL at the target sentence after 
                                        # viewing only the interrupting 
                                        # sentences (local context)
        ppl_glo_unintrp_all = defaultdict(list)
        ppl_glo_intrp_all = defaultdict(list)
        entropy_base_all = defaultdict(list)
        entropy_loc_unintrp_all = defaultdict(list)
        entropy_loc_intrp_all = defaultdict(list)
        entropy_glo_unintrp_all = defaultdict(list)
        entropy_glo_intrp_all = defaultdict(list)

        # SIGNAL
        ppl_loc_intrp_SIGNAL_all = defaultdict(list)
        ppl_glo_intrp_SIGNAL_all = defaultdict(list)
        entropy_loc_intrp_SIGNAL_all = defaultdict(list)
        entropy_glo_intrp_SIGNAL_all = defaultdict(list)

        for idx in tqdm(range(len(sents))):
            sent = sents[idx]

            if idx == 0:
                out, hid = model(sent, hid_init)
                continue

            # hid contains the cummulative context of the intact text
            _, _, hid, _ = sent_measurements(sent, model, vocab, hid)

            if idx + 1 not in intrp_inds:
                continue

            # NOTE: Reach the interruption point (idx=last sentence before intrp)
            preceding_all[idx] = sents_text[idx]
            hid_glo_intrp_all = []  # hidden states after viewing the intrp sents
                                    # and the global context

            hid_loc_intrp_all = []  # hidden states after viewing only the intrp
                                #   sents without any preceding global context

            """ With signal """
            hid_glo_intrp_SIGNAL_all = []  # hidden states after viewing the intrp sents
                                    # and the global context

            hid_loc_intrp_SIGNAL_all = []  # hidden states after viewing only the intrp
                                #   sents without any preceding global context

            hid_loc_unintrp = hid_init

            hid_glo_unintrp = hid

            generated_unintrp_all[idx+1].append(generate_text(model,vocab,hid_glo_unintrp,GENERATE_LENGTH))


            # NOTE: Feed in Interrupting Sentences
            for sim_level in SIM_RANGE:
                sents_intrp, scores_intrp = intrp_data[(idx + 1, sim_level)]

                # logging sents info
                sents_intrp_all[idx+1].append(sents_intrp)
                sim_scores_all[idx+1].append(scores_intrp)

                """ With signal """
                sents_intrp_SIGNAL = ["<beginning of unrelated> " + sents_intrp + " <end of unrelated>"]
                _, sents_intrp_SIGNAL = word2idx(sents_intrp_SIGNAL, vocab)  # embedding
                sents_intrp_SIGNAL = torch.cat(sents_intrp_SIGNAL)
                
                # contains S_global, S_intrp
                _, _, hid_glo_intrp_SIGNAL, _ = sent_measurements(
                    sents_intrp_SIGNAL, model, vocab, hid_glo_unintrp)
                hid_glo_intrp_SIGNAL_all.append(hid_glo_intrp_SIGNAL)

                # contains S_intrp only
                _, _, hid_loc_intrp_SIGNAL, _ = sent_measurements(
                    sents_intrp_SIGNAL, model, vocab, hid_init)
                hid_loc_intrp_SIGNAL_all.append(hid_loc_intrp_SIGNAL)

                """ Without signal"""
                _, sents_intrp = word2idx(sents_intrp, vocab)  # embedding
                sents_intrp = torch.cat(sents_intrp)

                # contains S_global, S_intrp
                _, _, hid_glo_intrp, _ = sent_measurements(
                    sents_intrp, model, vocab, hid_glo_unintrp)
                hid_glo_intrp_all.append(hid_glo_intrp)

                # Collecting generated text after intrp
                generated_intrp_all[idx+1].append(generate_text(model,vocab,hid_glo_intrp,GENERATE_LENGTH))

                # contains S_intrp only
                _, _, hid_loc_intrp, _ = sent_measurements(
                    sents_intrp, model, vocab, hid_init)
                hid_loc_intrp_all.append(hid_loc_intrp)

            # NOTE: Feed in Local Context
            ## (Skip if TARGET_DISTANCE=0 : targets are immediate)
            if TARGET_DISTANCE > 0:
                sents_loc = torch.cat(sents[idx+1 : idx+1+TARGET_DISTANCE])
                
                # contains S_global, S_local
                _, _, hid_glo_unintrp, _ = sent_measurements(
                    sents_loc, model, vocab, hid_glo_unintrp)

                # contains S_local only
                _, _, hid_loc_unintrp, _ = sent_measurements(
                    sents_loc, model, vocab, hid_init)

                for sim_level in SIM_RANGE:
                    """ Without signal"""
                    # contains S_global, S_intrp, S_local
                    _, _, hid_glo_intrp, _ = sent_measurements(
                        sents_loc, model, vocab, hid_glo_intrp_all[sim_level])
                    hid_glo_intrp_all[sim_level] = hid_glo_intrp

                    # contains S_intrp, S_local
                    _, _, hid_loc_intrp, _ = sent_measurements(
                        sents_loc, model, vocab, hid_loc_intrp_all[sim_level])
                    hid_loc_intrp_all[sim_level] = hid_loc_intrp

                    """ With signal"""
                    # contains S_global, S_intrp, S_local
                    _, _, hid_glo_intrp_SIGNAL, _ = sent_measurements(
                        sents_loc, model, vocab, hid_glo_intrp_SIGNAL_all[sim_level])
                    hid_glo_intrp_SIGNAL_all[sim_level] = hid_glo_intrp_SIGNAL

                    # contains S_intrp, S_local
                    _, _, hid_loc_intrp_SIGNAL, _ = sent_measurements(
                        sents_loc, model, vocab, hid_loc_intrp_SIGNAL_all[sim_level])
                    hid_loc_intrp_SIGNAL_all[sim_level] = hid_loc_intrp_SIGNAL

            # NOTE: Feed in Target and log PPL
            target_idx = idx + 1 + TARGET_DISTANCE
            sent_target = sents[target_idx]
            target_sent_all[target_idx] = sents_text[target_idx]

            # PPL ( S_target )
            ppl_base, _, _, entropy_base = sent_measurements(sent_target, model, vocab, hid_init)
            ppl_base_all[target_idx] = ppl_base.item()
            entropy_base_all[target_idx] = entropy_base.item()
            
            # PPL ( S_target | S_global, S_local )
            ppl_glo_unintrp, _, hid_glo_unintrp, entropy_glo_unintrp = sent_measurements(
                sent_target, model, vocab, hid_glo_unintrp)
            ppl_glo_unintrp_all[target_idx] = ppl_glo_unintrp.item()
            entropy_glo_unintrp_all[target_idx] = entropy_glo_unintrp.item()

            # PPL ( S_target | S_local )
            ppl_loc_unintrp, _, hid_loc_unintrp, entropy_loc_unintrp = sent_measurements(
                sent_target, model, vocab, hid_loc_unintrp)
            ppl_loc_unintrp_all[target_idx] = ppl_loc_unintrp.item()
            entropy_loc_unintrp_all[target_idx] = entropy_loc_unintrp.item()

            for sim_level in SIM_RANGE:
                # PPL ( S_target | S_global, S_intrp, S_local )
                ppl_glo_intrp, _, hid_glo_intrp, entropy_glo_intrp = sent_measurements(
                    sent_target, model, vocab, hid_glo_intrp_all[sim_level])
                ppl_glo_intrp_all[idx].append(ppl_glo_intrp.item())
                entropy_glo_intrp_all[idx].append(entropy_glo_intrp.item())

                # # PPL ( S_target | S_intrp, S_local )
                ppl_local_intrp, _, hid_loc_intrp, entropy_local_intrp = sent_measurements(
                    sent_target, model, vocab, hid_loc_intrp_all[sim_level])
                ppl_loc_intrp_all[idx].append(ppl_local_intrp.item())
                entropy_loc_intrp_all[idx].append(entropy_local_intrp.item())

                # Note: SIGNAL
                # PPL ( S_target | S_global, S_intrp_SIGNAL, S_local )
                ppl_glo_intrp_SIGNAL, _, hid_glo_intrp_SIGNAL, entropy_glo_intrp_SIGNAL = sent_measurements(
                    sent_target, model, vocab, hid_glo_intrp_SIGNAL_all[sim_level])
                ppl_glo_intrp_SIGNAL_all[idx].append(ppl_glo_intrp_SIGNAL.item())
                entropy_glo_intrp_SIGNAL_all[idx].append(entropy_glo_intrp_SIGNAL.item())

                # # PPL ( S_target | S_intrp_SIGNAL, S_local )
                ppl_local_intrp_SIGNAL, _, hid_loc_intrp_SIGNAL, entropy_local_intrp_SIGNAL = sent_measurements(
                    sent_target, model, vocab, hid_loc_intrp_SIGNAL_all[sim_level])
                ppl_loc_intrp_SIGNAL_all[idx].append(ppl_local_intrp_SIGNAL.item())
                entropy_loc_intrp_SIGNAL_all[idx].append(entropy_local_intrp_SIGNAL.item())

        # save data
        # Notes: all dicts are sorted since items are inserted as the original
        #        text is traversed. Need to explicitly sort the dict if the
        #        implementation changes later
        results = pd.concat([
            pd.Series(preceding_all.values()),
            pd.Series(target_sent_all.keys()), 
            pd.Series(target_sent_all.values()), 
            pd.Series(sents_intrp_all.values()),
            pd.Series(sim_scores_all.values()), 

            pd.Series(generated_unintrp_all.values()),
            pd.Series(generated_intrp_all.values()),

            pd.Series(ppl_base_all.values()), 
            pd.Series(ppl_loc_unintrp_all.values()),  
            pd.Series(ppl_loc_intrp_all.values()),
            pd.Series(ppl_glo_unintrp_all.values()),  
            pd.Series(ppl_glo_intrp_all.values()),
            pd.Series(entropy_base_all.values()), 
            pd.Series(entropy_loc_unintrp_all.values()),  
            pd.Series(entropy_loc_intrp_all.values()),
            pd.Series(entropy_glo_unintrp_all.values()),  
            pd.Series(entropy_glo_intrp_all.values()),

            pd.Series(ppl_loc_intrp_SIGNAL_all.values()),
            pd.Series(ppl_glo_intrp_SIGNAL_all.values()),
            pd.Series(entropy_loc_intrp_SIGNAL_all.values()),
            pd.Series(entropy_glo_intrp_SIGNAL_all.values())
        ],
        keys=[
            "preceding_sent",
            "target_idx", 
            "target_sent", 
            "sents_intrp_all_bins",
            "sim_intrp_all_bins",

            "generated_unintrp",
            "generated_intrp",

            "base_PPL",
            "local_unintrp_PPL",
            "local_intrp_PPL_all_bins",
            "global_unintrp_PPL",
            "global_intrp_PPL_all_bins",
            "base_entropy",
            "local_unintrp_entropy",
            "local_intrp_entropy_all_bins",
            "global_unintrp_entropy",
            "global_intrp_entropy_all_bins",

            "local_intrp_SIGNAL_PPL_all_bins",
            "global_intrp_SIGNAL_PPL_all_bins",
            "local_intrp_SIGNAL_entropy_all_bins",
            "global_intrp_SIGNAL_entropy_all_bins"
        ],
        axis=1)

        if PARAPHRASE: 
            saved_folder = os.path.join(result_dir, 
                f"{intrp_sent_num}-paraphrase_Target-{TARGET_DISTANCE+1}/")
        else:
            saved_folder = os.path.join(result_dir, 
                f"{intrp_sent_num}-sentence_Target-{TARGET_DISTANCE+1}/")
        saved_file_name = \
            f'ppl_LSTM_{model_size}_{target_type}.csv'
        create_folders_if_necessary(saved_folder)

        results.to_csv(os.path.join(saved_folder, saved_file_name), index=False,encoding='utf-8')

    # Run experiments
    for model in models:
        model_size = model.nhid
        # saving path
        result_dir = f'./results/LSTM-{model_size}'
        print(f"Model {model_size}")

        run("story", model, model_size, vocab, intrp_sent_num=INTRP_SENT_NUM)