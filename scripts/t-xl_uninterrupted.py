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

## Target sentences
target_sent_story_all = []
target_sent_story_plus_context_uninterrupted_all = []

target_sent_article_all = []
target_sent_article_plus_context_uninterrupted_all = []

### primary text: story ###

for counter_target in range(len(story_text)):

  target_sent_story_ = '<|endoftext|>' + story_text[counter_target]
  target_sent_story_all.append(target_sent_story_)
  
  target_sent_story_plus_context = ''

  for counter_sent_context_story in range(counter_target):
      target_sent_story_plus_context += story_text[counter_sent_context_story]

  target_sent_story_plus_context += target_sent_story_

  target_sent_story_plus_context_uninterrupted_all.append(target_sent_story_plus_context)


### primary text: article ###

for counter_target in range(len(article_text)):

  target_sent_article_ = '<|endoftext|>' + article_text[counter_target]
  target_sent_article_all.append(target_sent_article_)
  
  target_sent_article_plus_context = ''

  for counter_sent_context_article in range(counter_target):
      target_sent_article_plus_context += article_text[counter_sent_context_article]

  target_sent_article_plus_context += target_sent_article_

  target_sent_article_plus_context_uninterrupted_all.append(target_sent_article_plus_context)

## article
ppl_target_sent_base_article_all = []
ppls_target_sent_plus_context_article_all_uninterrupted = []

len_fix_win_tokens = 1024

context_len_article_all_uninterrupted = []

for counter_target in range(len(article_text)):
# for counter_target in range(3):

    print('counter_target: ' + str(counter_target))
    
    target_sent_article_ = target_sent_article_all[counter_target]
    target_sent_plus_context_article_ = target_sent_article_plus_context_uninterrupted_all[counter_target]

    lls_target_sent_article_ = []
    lls_target_sent_plus_context_article_ = []

    input_target_sent_tokenized_article_ = tokenizer_XL(target_sent_article_, return_tensors="pt")
    input_target_sent_plus_context_tokenized_article_ = tokenizer_XL(target_sent_plus_context_article_, return_tensors="pt")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    input_target_sent_tokenized_article_ = input_target_sent_tokenized_article_.to(device)
    input_target_sent_plus_context_tokenized_article_ = input_target_sent_plus_context_tokenized_article_.to(device)


    stride = 1
    
    
    #### base PPL ####
    
    for i in tqdm(range(1, input_target_sent_tokenized_article_.input_ids.size(1), stride)):
        begin_loc = max(i + stride - len_fix_win_tokens, 0)
        end_loc = min(i + stride, input_target_sent_tokenized_article_.input_ids.size(1))
        # print('end_loc wo C: '+str(end_loc))
        trg_len = end_loc - i    # may be different from stride on last loop
        # trg_len = len(input_target_sent_tokenized)
        input_ids = input_target_sent_tokenized_article_.input_ids[:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100


        with torch.no_grad():

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            outputs = model_XL(input_ids, labels=target_ids)
#             log_likelihood = outputs[0] * trg_len
            log_likelihood = outputs.losses[0][0]

        lls_target_sent_article_.append(log_likelihood)

    len_sent_base = len(lls_target_sent_article_[5:]) # reading from 5th element onward because to exclude lls of tokens from '<|endoftext|>'

    ppl_target_sent_base_article_all.append((torch.exp(torch.stack(lls_target_sent_article_[5:]).sum() / len_sent_base)).item())



    #### Plus context PPL ####

    for i in tqdm(range(1, input_target_sent_plus_context_tokenized_article_.input_ids.size(1), stride)):
        begin_loc = max(i + stride - len_fix_win_tokens, 0)
        end_loc = min(i + stride, input_target_sent_plus_context_tokenized_article_.input_ids.size(1))
        # print('end_loc wo C: '+str(end_loc))
        trg_len = end_loc - i    # may be different from stride on last loop
        # trg_len = len(input_target_sent_tokenized)
        input_ids = input_target_sent_plus_context_tokenized_article_.input_ids[:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            outputs = model_XL(input_ids, labels=target_ids)
#             log_likelihood = outputs[0] * trg_len
            log_likelihood = outputs.losses[0][0]

        lls_target_sent_plus_context_article_.append(log_likelihood)

    ppls_target_sent_plus_context_article_all_uninterrupted.append((torch.exp(torch.stack(lls_target_sent_plus_context_article_[-len_sent_base:]).sum() / len_sent_base)).item())
    context_len = end_loc - begin_loc
    context_len_article_all_uninterrupted.append(context_len)

## Saving
import pandas as pd
## saving target sentences and interruption sentneces ##
target_sent_article_all_ = pd.DataFrame(target_sent_article_all)

## saving context length ##
context_len_article_all_uninterrupted_ = pd.DataFrame(context_len_article_all_uninterrupted)

## saving ppl ##
ppl_target_sent_base_article_all_ = pd.DataFrame(ppl_target_sent_base_article_all)
ppls_target_sent_plus_context_article_all_uninterrupted_ = pd.DataFrame(ppls_target_sent_plus_context_article_all_uninterrupted)


PPL_results_article_all = pd.concat([target_sent_article_all_, context_len_article_all_uninterrupted_, ppl_target_sent_base_article_all_, ppls_target_sent_plus_context_article_all_uninterrupted_], keys = ['target_sent', 'context_len_uninterr', 'PPL_base', 'PPL_w_context_uninterr'], axis = 1)

PPL_results_article_all.to_csv('mem_400_UninterruptedPPL_tXL_APAMarriageArticle_FixLenWin_'+str(len_fix_win_tokens)+'.csv', index=False)

## Story
# ppl_target_sent_base_story_all = []
# ppls_target_sent_plus_context_story_all_uninterrupted = []

# len_fix_win_tokens = 1024

# context_len_story_all_uninterrupted = []

# start_run = 200
# end_run = 250

# for counter_target in range(start_run, end_run):
# # for counter_target in range(3):

#     print('counter_target: ' + str(counter_target))
    
#     target_sent_story_ = target_sent_story_all[counter_target]
#     target_sent_plus_context_story_ = target_sent_story_plus_context_uninterrupted_all[counter_target]

#     lls_target_sent_story_ = []
#     lls_target_sent_plus_context_story_ = []

#     input_target_sent_tokenized_story_ = tokenizer_XL(target_sent_story_, return_tensors="pt")
#     input_target_sent_plus_context_tokenized_story_ = tokenizer_XL(target_sent_plus_context_story_, return_tensors="pt")

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     input_target_sent_tokenized_story_ = input_target_sent_tokenized_story_.to(device)
#     input_target_sent_plus_context_tokenized_story_ = input_target_sent_plus_context_tokenized_story_.to(device)


#     stride = 1
    
    
#     #### base PPL ####
    
#     for i in tqdm(range(1, input_target_sent_tokenized_story_.input_ids.size(1), stride)):
#         begin_loc = max(i + stride - len_fix_win_tokens, 0)
#         end_loc = min(i + stride, input_target_sent_tokenized_story_.input_ids.size(1))
#         # print('end_loc wo C: '+str(end_loc))
#         trg_len = end_loc - i    # may be different from stride on last loop
#         # trg_len = len(input_target_sent_tokenized)
#         input_ids = input_target_sent_tokenized_story_.input_ids[:,begin_loc:end_loc]
#         target_ids = input_ids.clone()
#         target_ids[:,:-trg_len] = -100


#         with torch.no_grad():

#             input_ids = input_ids.to(device)
#             target_ids = target_ids.to(device)
#             outputs = model_XL(input_ids, labels=target_ids)
# #             log_likelihood = outputs[0] * trg_len
#             log_likelihood = outputs.losses[0][0]

#         lls_target_sent_story_.append(log_likelihood)

#     len_sent_base = len(lls_target_sent_story_[5:]) # reading from 5th element onward because to exclude lls of tokens from '<|endoftext|>'

#     ppl_target_sent_base_story_all.append((torch.exp(torch.stack(lls_target_sent_story_[5:]).sum() / len_sent_base)).item())



#     #### Plus context PPL ####

#     for i in tqdm(range(1, input_target_sent_plus_context_tokenized_story_.input_ids.size(1), stride)):
#         begin_loc = max(i + stride - len_fix_win_tokens, 0)
#         end_loc = min(i + stride, input_target_sent_plus_context_tokenized_story_.input_ids.size(1))
#         # print('end_loc wo C: '+str(end_loc))
#         trg_len = end_loc - i    # may be different from stride on last loop
#         # trg_len = len(input_target_sent_tokenized)
#         input_ids = input_target_sent_plus_context_tokenized_story_.input_ids[:,begin_loc:end_loc]
#         target_ids = input_ids.clone()
#         target_ids[:,:-trg_len] = -100

#         with torch.no_grad():

#             input_ids = input_ids.to(device)
#             target_ids = target_ids.to(device)
#             outputs = model_XL(input_ids, labels=target_ids)
# #             log_likelihood = outputs[0] * trg_len
#             log_likelihood = outputs.losses[0][0]

#         lls_target_sent_plus_context_story_.append(log_likelihood)

#     ppls_target_sent_plus_context_story_all_uninterrupted.append((torch.exp(torch.stack(lls_target_sent_plus_context_story_[-len_sent_base:]).sum() / len_sent_base)).item())
#     context_len = end_loc - begin_loc
#     context_len_story_all_uninterrupted.append(context_len)

# ##Saving
# import pandas as pd

# ## saving target sentences and interruption sentneces ##
# target_sent_story_all_ = pd.DataFrame(target_sent_story_all)

# ## saving context length ##
# context_len_story_all_uninterrupted_ = pd.DataFrame(context_len_story_all_uninterrupted)

# ## saving ppl ##
# ppl_target_sent_base_story_all_ = pd.DataFrame(ppl_target_sent_base_story_all)
# ppls_target_sent_plus_context_story_all_uninterrupted_ = pd.DataFrame(ppls_target_sent_plus_context_story_all_uninterrupted)


# PPL_results_story_all = pd.concat([target_sent_story_all_, context_len_story_all_uninterrupted_, ppl_target_sent_base_story_all_, ppls_target_sent_plus_context_story_all_uninterrupted_], keys = ['target_sent', 'context_len_uninterr', 'PPL_base', 'PPL_w_context_uninterr'], axis = 1)

# PPL_results_story_all.to_csv('mem_400_UninterruptedPPL_tXL_RainyDayStory_FixLenWin_'+str(len_fix_win_tokens)+'_'+str(start_run)+'_'+str(end_run)+'.csv', index=False)
