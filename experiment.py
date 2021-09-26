import pandas as pd
import numpy as np
import ast
import os

from utils.data import preprocess, word2idx

# Hyperparams for now
TARGET_NUM = 5  # number of targets to select
TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence

# data path
data_dir = './data'
story_path = os.path.join(data_dir, 
    "text",
    'On a Rainy Day - by Pat Garcia.docx')
article_path = os.path.join(data_dir, 
    "text",
    'Can this marriage be saved -  APA Sicence Watch.docx')

story_pool_path = os.path.join(data_dir,
    "pools",
    'interruption_sim_RainyDayStory_pool_brown_allCatges_seed_1.xlsx')

article_pool_path = os.path.join(data_dir,
    'pools',
    'interruption_sim_APAMarriageArticle_pool_brown_allCatges_seed_1.xlsx')

# print(target)
# print(article_pool.loc[article_pool["target_sent"] == "Ask any young \
#             couple how long their marriage will last, and chances\
#             are, they'll say forever, says Clark University psychologist \
#             Jeffrey Jensen Arnett, PhD, an expert on emerging adulthood."])


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
        assert(sents[idx]==target_sent)

        low_sim_sent = pool['intrp_sentence_low_sim'][idx]
        # QS: 0-based or 1-based?
        low_sim_sent = ast.literal_eval(low_sim_sent)[seed_num-1]
        low_sim_score = pool['sim_score_low'][idx]
        low_sim_score = ast.literal_eval(low_sim_score)[seed_num-1]

        mid_sim_sent = pool['intrp_sentence_mid_sim'][idx]
        mid_sim_sent = ast.literal_eval(mid_sim_sent)[seed_num-1]
        mid_sim_score = pool['sim_score_mid'][idx]
        mid_sim_score = ast.literal_eval(mid_sim_score)[seed_num-1]

        high_sim_sent = pool['intrp_sentence_high_sim'][idx]
        high_sim_sent = ast.literal_eval(high_sim_sent)[seed_num-1]
        high_sim_score = pool['sim_score_high'][idx]
        high_sim_score = ast.literal_eval(high_sim_score)[seed_num-1]
    
        # interruption data as a dictionary
        # {(target_idx, sim_level) : (intrp_sent, score)}
        # where sim_level ranges from 1-10
        
        intrp_data[(idx, 0)] = (low_sim_sent, low_sim_score)
        intrp_data[(idx, 1)] = (mid_sim_sent, mid_sim_score)
        intrp_data[(idx, 2)] = (high_sim_sent, high_sim_score)


    return target_inds, intrp_data

def run(target_type, model, seed_num):
    return -1

target_inds_story, intrp_data_story =  prepare_input('story',1)
print(intrp_data_story)