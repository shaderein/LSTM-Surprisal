import pandas as pd
import numpy as np

from .data import preprocess, word2idx

# Hyperparams for now
TARGET_NUM = 5  # number of targets to select
TARGET_IDX_LOW = 6  # start selecting target from the n-th sentence

# data path
story_path = "data/text/On a Rainy Day - by Pat Garcia.docx"
article_path = "data/text/Can this marriage be saved -  \
                APA Sicence Watch.docx"

story_pool_path = 'data/pools/interruption_sim_RainyDayStory_pool_\
                brown_allCatges_seed_1.xlsx'
article_pool_path = 'data/pools/interruption_sim_APAMarriageArticle_\
                pool_brown_allCatges_seed_1.xlsx'

story_pool = pd.read_excel(story_pool_path,
                           header=0,
                           converters={'target_sent': list})
article_pool = pd.read_excel(article_pool_path,
                             converters={'target_sent': list})

target = article_pool.loc[0]
print(target)
print(article_pool.loc[article_pool["target_sent"] == "Ask any young \
            couple how long their marriage will last, and chances\
            are, they'll say forever, says Clark University psychologist \
            Jeffrey Jensen Arnett, PhD, an expert on emerging adulthood."])


def prepare_input(sents, pool_path, seed_num):
    np.random.seed(seed_num)
    target_inds = np.random.randint(TARGET_IDX_LOW, len(sents), TARGET_NUM)
    target_sents = sents[target_inds]
    
    # interruption data as a dictionary
    # {(target_idx, sim_level) : (intrp_sent, score)}
    # where sim_level ranges from 1-10
    intrp_data = {}


    return target_inds, target_sents, intrp_data

def run(target_type="story", model, seed_num):
    return -1