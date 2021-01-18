
#basics
import random
import pandas as pd
import torch
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from nltk import TreebankWordTokenizer
from collections import Counter
from venn import venn


class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def create_data_list(self, filename_list):
        #return two lists, create id2word and id2ner mapping dicts
        
        data_list = []
        ner_list = []
        self.id2word = {}
        self.id2ner = {}
        ner_id = 1
        word_id = 1

        puncts= "()-,.?!:;*/--"
        
        for filename in filename_list:
            #split train and validation dataset
            if 'Test' in str(filename):
                split = 'test'
            else:
                split = random.choices(["train", "val"], weights = (80, 20), k = 1)[0]  # split train into train 
            
            #parse xml data
            tree = ET.parse(filename)
            root = tree.getroot()
            for elem in root:
                sent_id = elem.get("id")
                sentence = elem.get("text")
                text_tokens = TreebankWordTokenizer().tokenize(sentence)
                text_tokenized = [word.strip(puncts).lower() if word[-1] in puncts else word for word in text_tokens]
                text_tokenized = list(filter(None, text_tokenized)) 
                span_text = list(TreebankWordTokenizer().span_tokenize(sentence)) 
                
                # creat data list            
                char_ids = []
                for st in span_text:
                    char_ids.append((st[0], (st[1]-1)))
                for i, token in enumerate(text_tokenized):
                    if token.lower() not in self.id2word.values():
                        self.id2word[word_id] = token.lower()
                        word_id += 1
                    for id, word in self.id2word.items():
                        if word == token.lower():
                            token_id = id
                    word_info_list = (sent_id, token_id, int(char_ids[i][0]), int(char_ids[i][1]), split)
                    data_list.append(word_info_list)
                  
                # creat NER data list             
                for sub_elem in elem:
                    if sub_elem.tag == "entity":
                        ner = sub_elem.get("type")
                        if ner not in self.id2ner.values():
                            self.id2ner[ner_id] = ner
                            ner_id += 1
                        for id, ner_tmp in self.id2ner.items():
                            if ner_tmp == ner:
                                label = id
                        #get char_start_id and char_end_id
                        if ";" not in sub_elem.get("charOffset"):
                            char_start, char_end = sub_elem.get("charOffset").split("-")
                            char_start, char_end = int(char_start), int(char_end)
                            ner_list.append([sent_id, label, char_start, char_end])
                        #if more than one mention of an entity, split into several lines
                        else:
                            occurences = sub_elem.get("charOffset").split(";")
                            for occurence in occurences:
                                char_start, char_end = occurence.split("-")
                                char_start, char_end = int(char_start), int(char_end)
                                ner_list.append([sent_id, label, char_start, char_end])

        self.vocab = list(self.id2word.values())
        return data_list, ner_list


    def get_max_length(self):
        #gets the length of the longest sentence 
        sentences = list(self.data_df["sentence_id"])
        word_count = Counter(sentences)
        max_l = max(list(word_count.values()))
                
        return max_l


    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
 
        all_filename_list = [f for f in Path(data_dir).glob('**/*.xml')]
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.

        # TODO
        # parsing XML files in Training and testing folders
        data_list, ner_list = self.create_data_list(all_filename_list)
        
        # creating 2 dataframes: data_df ner_df
            # data_df
                # sentence_id, token_id, char_start_id, char_end_id, split(Train, Val, Test)
                # Split xml to different categories
                # tokenization
                # create token_id dict, starting from 1
                # calculating char_start,char_end, filter punctuation
        self.data_df = pd.DataFrame(data_list, columns=["sentence_id", "token_id", "char_start_id", "char_end_id", "split"])

            # ner_df
                # sentence_id, ner_id, char_start_id, char_end_id
                # create ner_id dict, starting from 1 in entity tag
        self.ner_df = pd.DataFrame(ner_list, columns=["sentence_id", "ner_id", "char_start_id", "char_end_id"])
        self.max_sample_length = self.get_max_length()


    def get_labels_from_ner_df(self, df): 
        #takes a dataframe and returns a list of all ner labels (devidable by the max_sample_length)
    
        label_list = []
        all_labels = []
        
        sent_ids = [s for s in df["sentence_id"]]
        start_ids = [s for s in df["char_start_id"]]
        end_ids = [s for s in df["char_end_id"]]
        id_tuples = list(zip(sent_ids, start_ids, end_ids))
        
        label_sent_ids = [s for s in self.ner_df["sentence_id"]]
        label_start_ids = [s for s in self.ner_df["char_start_id"]]
        label_end_ids = [s for s in self.ner_df["char_end_id"]]
        labels = [s for s in self.ner_df["ner_id"]]
        label_tuples = list(zip(label_sent_ids, label_start_ids, label_end_ids))
        
        if sent_ids:
            sentence = sent_ids[0]
        else: 
            sentence = 0
        sent_labels = []
        for t in id_tuples:
            label = 0
            if t in label_tuples:
                label = labels[label_tuples.index(t)]
            if t[0] == sentence:
                sent_labels.append(label)
            else:
                diff = self.max_sample_length - len(sent_labels)
                padding = diff * [-1]
                sent_labels.extend(padding)
                label_list.append(sent_labels[:self.max_sample_length])
                sent_labels = [label]
                sentence = t[0]
            if label != 0:
                all_labels.append(label) 
               
        return label_list, all_labels


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        
        #prepare data for plotting and labeling
        #split the data_df into three sub df: train, val and test
        val_df = self.data_df.loc[self.data_df['split'] == 'val']
        train_df = self.data_df.loc[self.data_df['split'] == 'train']
        test_df = self.data_df.loc[self.data_df['split'] == 'test']
        
        #get labels for each of the split dfs and shape into the correct dimensions
        self.train_list, self.all_labels_train = self.get_labels_from_ner_df(train_df)
        self.train_tensor_y = torch.LongTensor(self.train_list)
        self.train_tensor_y = self.train_tensor_y.to(self.device)
        
        self.val_list, self.all_labels_val = self.get_labels_from_ner_df(val_df)
        self.val_tensor_y = torch.LongTensor(self.val_list)
        self.val_tensor_y = self.val_tensor_y.to(self.device)
        
        self.test_list, self.all_labels_test = self.get_labels_from_ner_df(test_df)
        self.test_tensor_y = torch.LongTensor(self.test_list)
        self.test_tensor_y = self.test_tensor_y.to(self.device)
        
        return self.train_tensor_y, self.val_tensor_y, self.test_tensor_y


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        self.get_y()
        
        train_id_count = Counter(self.all_labels_train)
        train_ner_count = {}
        for key in train_id_count.keys():
            train_ner_count[self.id2ner[key]] = train_id_count[key]

        val_id_count = Counter(self.all_labels_val)
        val_ner_count = {}
        for key in val_id_count.keys():
            val_ner_count[self.id2ner[key]] = val_id_count[key]

        test_id_count = Counter(self.all_labels_test)
        test_ner_count = {}
        for key in test_id_count.keys():
            test_ner_count[self.id2ner[key]] = test_id_count[key]

        data = [train_ner_count, val_ner_count, test_ner_count]
        to_plot= pd.DataFrame(data,index=['train', 'val', 'test'])
        to_plot.plot.bar(figsize=(5,10))
        plt.show()


    def plot_ner_per_sample_distribution(self):        
    # FOR BONUS PART!!
    # Should plot a histogram displaying the distribution of number of NERs in sentences
    # e.g. how many sentences has 1 ner, 2 ner and so on
        print('Distribution of number of NERs in sentences')
        
        counter_dict= {}
        sentence_ids = list(self.data_df["sentence_id"].unique())
        for sentence_id in sentence_ids:
            sub_ner_df = self.ner_df.loc[self.ner_df['sentence_id'] == sentence_id]
            count = len(sub_ner_df.index)
            if count not in counter_dict.keys():
                counter_dict[count] = [sentence_id]
            else:
                if sentence_id not in counter_dict[count]:
                    counter_dict[count].append(sentence_id)
        keys = list(counter_dict.keys())
        data = [len(sentences) for sentences in counter_dict.values()]
        keys.sort()
        to_plot= pd.DataFrame(data,index=keys)
        to_plot.plot.bar(figsize=(10,10))
        plt.show()


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        print('Distribution of sample lengths in number tokens')
        
        counter_dict= {}
        sentence_ids = list(self.data_df["sentence_id"].unique())
        for sentence in sentence_ids:
            sub_df = self.data_df.loc[self.data_df['sentence_id'] == sentence]
            count = len(sub_df.index)
            if count not in counter_dict.keys():
                counter_dict[count] = 1
            else:
                counter_dict[count] += 1      
        keys = list(counter_dict.keys())
        data = counter_dict.values()
        keys.sort()
        to_plot= pd.DataFrame(data,index=keys)
        to_plot.plot.bar(figsize=(20,5))
        plt.show()


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        venn_data = {}

        for ner_id in self.ner_df['ner_id'].unique():
            ner_entries = self.ner_df[self.ner_df['ner_id'] == ner_id]
            sentence_ids = ner_entries['sentence_id'].unique()

            venn_data[self.id2ner[ner_id]] =set(sentence_ids)

        to_plot = venn(venn_data)
        to_plot.set_title('NER labels co-occur in sentences')
        # plt = to_plot.get_figure()
        plt.show()



