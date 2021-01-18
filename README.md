# LT2316 H20 Assignment A1

Name: Fang Yuan

## Notes on Part 1.

I put the dataset `DDICorpus` folder at the same level path as `aa` folder. I created a class method `create_data_list` for `DataLoader` to parse xml files and build the dataframe `data_df` and `ner_df`.

### data_df

Firstly, I took 20% of the training dataset split and turned it into the validation set. Then I use `TreebankWordTokenizer` to tokenize the words in each sentence. I filter the punctuations and the empty strings in the tokenized words list.

### ner_df

I used the similar way to build the `ner_df` like I made for `data_df`.  I got 4 entities `group`, `drug_n`, `drug` and `brand` types. The entity `drug` has the largest number compared to other 3 entity types among all datasets.

## Notes on Part 2.

I chosed 3 features:

- Neighboring words. The context of a word is very important in defining the word itself. If one of the neighbor words is missing, I filtered this word in this feature list.
- POS-tags. I used `part-of-speech tagger` from NLTK library to processes words. This feature could be very useful to identify the entities in the sentences.
- Word length. This feature is designed to get the length of the drug names which could be an interesting feature for classifying NER label correctly.

## Notes on Part Bonus.

### plot_ner_per_sample_distribution

It seems like most of the sentences have 0~3 ners.

### plot_sample_length_distribution

The plot shows that the majority of sentences has 1~ 41 tokens.

### plot_ner_cooccurence_venndiagram

I use `venn` module to plot the NER labels co-occur in sentences.
