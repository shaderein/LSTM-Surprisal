# from transformers import TransfoXLConfig, TransfoXLModel
from transformers import TransfoXLModel
import torch
from transformers import pipeline
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
# from data import preprocess, word2idx, Dictionary
from nltk import sent_tokenize

### for Transformer-XL ###

# from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLConfig

configuration = TransfoXLConfig(mem_len=400)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer_XL = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model_XL = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103', config=configuration)
model_XL.to(device)

print(f"mem_len = {model_XL.config.mem_len}\n")

from utils.data import preprocess, word2idx, Dictionary
from nltk import sent_tokenize

article_file = './data/text/Can this marriage be saved -  APA Sicence Watch.docx'
story_file = './data/text/On a Rainy Day - by Pat Garcia.docx'

import nltk
nltk.download('punkt')

article_text, _ = preprocess(article_file)
story_text, _ = preprocess(story_file)

import nltk
nltk.download('brown')
import numpy as np
from nltk.corpus import brown

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

import pandas as pd
import ast

seed_num = 1

# TODO: use utils functions to load text and pools instead
intrp_pool_article = pd.read_excel("./data/pools/diverseSim_interruptions_APAMarriageArticle_pool_brown_allCatges_seed_"+str(seed_num)+".xlsx")

nums_run_seed_start = 4
nums_run_seed_end = 5

nums_trg_sent_1st = 5 # (5)

nums_sim_bin = 6 # (6) number of bins in the diverse similarity range

num_intrp_sents = 4

# debug_mode_len = 10
# len_fix_win_tokens = debug_mode_len

len_fix_win_tokens = 1024

stride = 1

for counter_run in range(nums_run_seed_start, nums_run_seed_end+1):

    print('** run_num: '+str(counter_run))

    target_sents_1st_run = [] # should save
    target_sents_2nd_run = [] # should save
    target_sents_3rd_run = [] # should save
    
    sents_intrp_bins = []
    sims_intrp_bins = []
    
    PPL_intrp_bins_2nd_trg = []
    context_len_intrp_bins_2nd_trg = []
    
    PPL_intrp_bins_3rd_trg = []
    context_len_intrp_bins_3rd_trg = []
    
    np.random.seed(counter_run)
    
    inds_target_sents_1st_ = np.random.randint(1, len(article_text)-2, nums_trg_sent_1st)
    
    for counter_trg_sent_1st in tqdm(range(nums_trg_sent_1st)):

        print('* counter_trg_sent_1st: '+str(counter_trg_sent_1st))
        
        ind_trg_sent_1st = inds_target_sents_1st_[counter_trg_sent_1st]
        
        target_sent_1st = '<|endoftext|>' + article_text[inds_target_sents_1st_[counter_trg_sent_1st]]
        target_sent_2nd = '<|endoftext|>' + article_text[(inds_target_sents_1st_[counter_trg_sent_1st])+1]
        target_sent_3rd = '<|endoftext|>' + article_text[(inds_target_sents_1st_[counter_trg_sent_1st])+2]
        
        if ('*' in target_sent_1st) or (len(target_sent_1st) < 20) or (target_sent_1st in target_sents_1st_run):
            
            ind_trg_sent_1st = np.random.randint(1, len(article_text)-2, 1)
            target_sent_1st = '<|endoftext|>' + article_text[ind_trg_sent_1st]
            target_sent_2nd = '<|endoftext|>' + article_text[ind_trg_sent_1st+1]
            target_sent_3rd = '<|endoftext|>' + article_text[ind_trg_sent_1st+2]

            inds_target_sents_1st_[counter_trg_sent_1st] = ind_trg_sent_1st

        target_sent_1st_text = target_sent_1st[13:]
        target_sent_2nd_text = target_sent_2nd[13:]
        target_sent_3rd_text = target_sent_3rd[13:]

        sent_intrp_bins_ = []
        sims_intrp_bins_ = []
        
        PPL_intrp_bins_2nd_trg_ = []
        lls_target2nd_plus_context_interrupted_bin_ = []
        context_len_intrp_bins_2nd_trg_ = []
        
        PPL_intrp_bins_3rd_trg_ = []
        lls_target3rd_plus_context_interrupted_bin_ = []
        context_len_intrp_bins_3rd_trg_ = []
        
        target_plus_context_article = ''    
        
        for counter_sent_context_article in range(ind_trg_sent_1st):
            
            target_plus_context_article += article_text[counter_sent_context_article]
            
        prior_context_unintrp = target_plus_context_article
        
        context_unintrp_tokenized = tokenizer_XL(prior_context_unintrp, return_tensors="pt")
        context_len_unintrp_ = len(context_unintrp_tokenized)

        for counter_sim_bin in range(1, nums_sim_bin+1):

            exec(f'sent_intrp_sim_bin = ast.literal_eval(intrp_pool_article["sents_sim_intrp_article_bin{counter_sim_bin}_all"][inds_target_sents_1st_[counter_trg_sent_1st]])[(counter_run*num_intrp_sents+1):(counter_run+1)*num_intrp_sents+1]')
            exec(f'sim_intrp_bin = ast.literal_eval(intrp_pool_article["sim_intrp_article_bin{counter_sim_bin}_all"][inds_target_sents_1st_[counter_trg_sent_1st]])[(counter_run*num_intrp_sents+1):(counter_run+1)*num_intrp_sents+1]')

            sent_intrp_concat_sim_bin = ''

            for i in range(len(sent_intrp_sim_bin)):
              sent_intrp_concat_sim_bin += sent_intrp_sim_bin[i]

            sent_intrp_bins_.append(sent_intrp_concat_sim_bin)
            sims_intrp_bins_.append(sim_intrp_bin)


            #### ****************** beginning of the modified section ******************

            exec(f'context_plus_intrp_bin{counter_sim_bin}_ = target_plus_context_article + sent_intrp_concat_sim_bin')
            exec(f'target_sent_1st_article_plus_context_intrp_bin{counter_sim_bin} = context_plus_intrp_bin{counter_sim_bin}_ + target_sent_1st')
            exec(f'target_sent_2nd_article_plus_context_intrp_bin{counter_sim_bin} = context_plus_intrp_bin{counter_sim_bin}_ + target_sent_1st_text + target_sent_2nd')
            exec(f'target_sent_3rd_article_plus_context_intrp_bin{counter_sim_bin} = context_plus_intrp_bin{counter_sim_bin}_ + target_sent_1st_text + target_sent_2nd_text + target_sent_3rd')

            #### ****************** end of the modified section ******************

        # print('1st : '+str(target_sent_1st_article_plus_context_intrp_bin1))
        # print('2nd : '+str(target_sent_2nd_article_plus_context_intrp_bin1))
        # print('3rd : '+str(target_sent_3rd_article_plus_context_intrp_bin1))

        ######### Calculating PPL of Interrupted Text #########
        
        for counter_sim_bin in range(1, nums_sim_bin+1):

            exec(f'input_target2nd_plus_context_interrupted_bin{counter_sim_bin}_tokenized = tokenizer_XL(target_sent_2nd_article_plus_context_intrp_bin{counter_sim_bin}, return_tensors="pt")')
            exec(f'input_target3rd_plus_context_interrupted_bin{counter_sim_bin}_tokenized = tokenizer_XL(target_sent_3rd_article_plus_context_intrp_bin{counter_sim_bin}, return_tensors="pt")')

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
            exec(f'input_target2nd_plus_context_interrupted_bin_tokenized = input_target2nd_plus_context_interrupted_bin{counter_sim_bin}_tokenized.to(device)')
            exec(f'input_target3rd_plus_context_interrupted_bin_tokenized = input_target3rd_plus_context_interrupted_bin{counter_sim_bin}_tokenized.to(device)')

            #### Interrupted-bin, plus context PPL , 2nd target ####
        
            input_target_sent2nd_tokenized_ = tokenizer_XL(target_sent_2nd, return_tensors="pt")
        
            len_base_target2nd = input_target_sent2nd_tokenized_.input_ids.size(1) - 6 ## subtrackting by 6 because the length of the input is 1 less than the output and 5 of lls elements are from the "enf of sentence" sign.

            for i in tqdm(range(1, input_target2nd_plus_context_interrupted_bin_tokenized.input_ids.size(1), stride)):
                
                begin_loc = max(i + stride - len_fix_win_tokens, 0)
                end_loc = min(i + stride, input_target2nd_plus_context_interrupted_bin_tokenized.input_ids.size(1))
                # print('end_loc wo C: '+str(end_loc))
                trg_len = end_loc - i    # may be different from stride on last loop
                # trg_len = len(input_target_sent_tokenized)
                input_ids = input_target2nd_plus_context_interrupted_bin_tokenized.input_ids[:,begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:,:-trg_len] = -100

                with torch.no_grad():

                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    outputs = model_XL(input_ids, labels=target_ids)
            #             log_likelihood = outputs[0] * trg_len
                    log_likelihood = outputs.losses[0][0]

                lls_target2nd_plus_context_interrupted_bin_.append(log_likelihood)

            PPL_intrp_bins_2nd_trg_.append((torch.exp(torch.stack(lls_target2nd_plus_context_interrupted_bin_[-len_base_target2nd:]).sum() / len_base_target2nd)).item())
            
            context_len_2nd_trg = end_loc - begin_loc
            context_len_intrp_bins_2nd_trg_.append(context_len_2nd_trg)


            #### Interrupted-bin, plus context PPL , 3rd target ####
        
            input_target_sent3rd_tokenized_ = tokenizer_XL(target_sent_3rd, return_tensors="pt")
        
            len_base_target3rd = input_target_sent3rd_tokenized_.input_ids.size(1) - 6 ## subtrackting by 6 because the length of the input is 1 less than the output and 5 of lls elements are from the "enf of sentence" sign.

            for i in tqdm(range(1, input_target3rd_plus_context_interrupted_bin_tokenized.input_ids.size(1), stride)):
                
                begin_loc = max(i + stride - len_fix_win_tokens, 0)
                end_loc = min(i + stride, input_target3rd_plus_context_interrupted_bin_tokenized.input_ids.size(1))
                # print('end_loc wo C: '+str(end_loc))
                trg_len = end_loc - i    # may be different from stride on last loop
                # trg_len = len(input_target_sent_tokenized)
                input_ids = input_target3rd_plus_context_interrupted_bin_tokenized.input_ids[:,begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:,:-trg_len] = -100

                with torch.no_grad():

                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    outputs = model_XL(input_ids, labels=target_ids)
            #             log_likelihood = outputs[0] * trg_len
                    log_likelihood = outputs.losses[0][0]

                lls_target3rd_plus_context_interrupted_bin_.append(log_likelihood)

            PPL_intrp_bins_3rd_trg_.append((torch.exp(torch.stack(lls_target3rd_plus_context_interrupted_bin_[-len_base_target3rd:]).sum() / len_base_target3rd)).item())
            
            context_len_3rd_trg = end_loc - begin_loc
            context_len_intrp_bins_3rd_trg_.append(context_len_3rd_trg)



        target_sents_1st_run.append(target_sent_1st)
        target_sents_2nd_run.append(target_sent_2nd)
        target_sents_3rd_run.append(target_sent_3rd)
        
        sents_intrp_bins.append(sent_intrp_bins_)
        sims_intrp_bins.append(sims_intrp_bins_)
        
        PPL_intrp_bins_2nd_trg.append(PPL_intrp_bins_2nd_trg_)   
        context_len_intrp_bins_2nd_trg.append(context_len_intrp_bins_2nd_trg_)
        
        PPL_intrp_bins_3rd_trg.append(PPL_intrp_bins_3rd_trg_)   
        context_len_intrp_bins_3rd_trg.append(context_len_intrp_bins_3rd_trg_)

    ### Saving the results for each run ###

    columns_ = ['target_sent_1st', 'sents_intrp_all_bins', 'sim_intrp_all_bins', 'target_sent_2nd', 'ppl_intrp_2ndtarget_all_bins', 'context_len_intrp_all_bins_2nd_trg', 'target_sent_3rd', 'ppl_intrp_3rdtarget_all_bins', 'context_len_intrp_all_bins_3rd_trg']

    intrp_ppl_results_all = []

    for counter_trg_sent_1st in range(nums_trg_sent_1st):

        intrp_ppl_result_sent = []

        intrp_ppl_result_sent.append([target_sents_1st_run[counter_trg_sent_1st]])
        intrp_ppl_result_sent.append(sents_intrp_bins[counter_trg_sent_1st])
        intrp_ppl_result_sent.append(sims_intrp_bins[counter_trg_sent_1st])
        
        intrp_ppl_result_sent.append([target_sents_2nd_run[counter_trg_sent_1st]])
        intrp_ppl_result_sent.append(PPL_intrp_bins_2nd_trg[counter_trg_sent_1st])
        intrp_ppl_result_sent.append(context_len_intrp_bins_2nd_trg[counter_trg_sent_1st])
        
        intrp_ppl_result_sent.append([target_sents_3rd_run[counter_trg_sent_1st]])
        intrp_ppl_result_sent.append(PPL_intrp_bins_3rd_trg[counter_trg_sent_1st])
        intrp_ppl_result_sent.append(context_len_intrp_bins_3rd_trg[counter_trg_sent_1st])

        intrp_ppl_results_all.append(intrp_ppl_result_sent)

    # saving result in xlsx format

    intrp_ppl_results_all = pd.DataFrame(intrp_ppl_results_all, columns=columns_)
    intrp_ppl_results_all.to_excel("mem_400_PPL_tXL_results_4sentsIntrp_MultipleTargets_2nd3rd_Debugged_v2_6bins_APA_marriage_FixLenWin_1024_seed_"+str(counter_run)+".xlsx") 

