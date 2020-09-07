
#basics
import random
import pandas as pd


class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str):
        self._parse_xml_data(data_dir)
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


    @property
    def max_sample_length(self):
        return self._max_sample_length

    @property
    def ner2id(self):
        return self._ner2id

    @property
    def id2ner(self):
        return self._id2ner

    @property
    def vocab(self):
        return list(self._id2word.values())

    @property
    def word2id(self):
        return self._word2id

    @property
    def id2word(self):
        return self._id2word

    @property
    def data_df(self):
        return self._data_df

    @property
    def ner_df(self):
        return self._ner_df


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


    def __init__(self, data_dir:str):
        super().__init__(data_dir=data_dir)


    def _parse_xml_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        pass


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        pass


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        pass


    def plot_sample_length_distribution(self):
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_sample_distribution(self):
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_ven_diagram(self):
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass


