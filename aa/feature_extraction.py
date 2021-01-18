
#basics
import pandas as pd
import torch
import nltk

def get_features(data, max_sample_length, id2word):
    
    sent_ids = [s for s in data["sentence_id"]]
    token_ids = [s for s in data["token_id"]]
    start_ids = [s for s in data["char_start_id"]]
    end_ids = [s for s in data["char_end_id"]]
    split = [s for s in data["split"]]
    all_rows = list(zip(sent_ids, token_ids, start_ids, end_ids, split))
    
    all_list = []
    sent_list = []

    pos2id_dict = {}
   

    for i in range(len(all_rows)):
        
        data_tuple = all_rows[i]
        sent_id = data_tuple[0]
        token_id = data_tuple[1]
        start_id = data_tuple[2]
        end_id = data_tuple[3]
        split = data_tuple[4]

        #first features: left and right neighbour in the sentence
        n_l = 0
        n_r = 0
        if i != 0:
            if all_rows[i-1][0] == sent_id:
                n_l = all_rows[i-1][1]
        if i < len(all_rows)-1:
            if all_rows[i+1][0] == sent_id:
                n_r = all_rows[i+1][1]
        
                
        #second feature: pos-tag
        word = id2word[token_id]
        pos_tag = nltk.tag.pos_tag(word)
        if pos_tag[0][1] not in pos2id_dict:
            pos2id_dict[pos_tag[0][1]] = len(pos2id_dict) + 1

        
        #third feature: word length
        word_len = end_id - start_id
        
        #feature list
        word_list = [n_l, n_r, pos2id_dict[pos_tag[0][1]], word_len]
        
        
        if i < len(all_rows)-1:
            if sent_id == all_rows[i+1][0]:
                sent_list.append(word_list)
            else:
                len_sent = len(sent_list)
                diff = max_sample_length - len_sent
                padding = diff * [[-1] * len(word_list)]
                sent_list.extend(padding)
                all_list.append(sent_list)
                sent_list = []
                sent_list.append(word_list)
        else:
            len_sent = len(sent_list)
            diff = max_sample_length - len_sent
            padding = diff * [[-1] * len(word_list)]
            sent_list.extend(padding)
            all_list.append(sent_list)
            sent_list = []
            sent_list.append(word_list)
            
    return torch.tensor(all_list)
       
    

def extract_features(data:pd.DataFrame, max_sample_length:int, id2word, device):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
   
    train_df = data.loc[data['split'] == 'train']
    train_tensor = get_features(train_df, max_sample_length, id2word)
    train_tensor = train_tensor.to(device)
    
    val_df = data.loc[data['split'] == 'val']
    val_tensor = get_features(val_df, max_sample_length, id2word)
    val_tensor = val_tensor.to(device)
    
    test_df = data.loc[data['split'] == 'test']
    test_tensor = get_features(test_df, max_sample_length, id2word)
    test_tensor = test_tensor.to(device)  

    return train_tensor, val_tensor, test_tensor
