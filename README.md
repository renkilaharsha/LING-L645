<h1 align="center">Multi-lingual Embedding Efficiency on ONET Data </h1>
<h3 align="center">Harsha Renkila</h3>


## ℹ️ Overview
Representation of Text in the N-Dimensional space is key for many NlP tasks such as NER, Sentiment analysis, Classisication, Language Modelling, etc.. has gain vital importance in the recent days. These representation of text as embeddings which captures the semantic meaning of the text. Most of the embeddings are training on the english corpus and showed capability of achieving NLP tasks.  When it comes to other languages there are different models for each language which can do the tasks. But, in realtime if a product support multiple languages let say 100, deploying 100 different language models for all languages will not highly resource taking and highly expensive. Due to that there are some Multilingual language models like Multilingual Bert, Multilingual Distill Bert, Multilingual Universal Sentence Encoders, Multilingual XLMR Bert, etc.. are supporting the more than 100 different languages with single model. These word embedding models are trained and evaluated with text with general context like wikipedia, books, etc..  using cosine-similarity. When it comes to domain-specific data, we want to evaluate the model without pre-training, how it performs across different languages on the same task.

## Languages Considered for Experiment
    - English (language Code - En)
    - Spanish (language Code - ES)
    - German (language Code - DE)
    - French (language Code - FR)
    - Dutch (language Code - NL)
## Dataset
### O*net
The ONET Program is the nation's primary source of occupational information. Valid data are essential to understanding the rapidly changing nature of work and how it impacts the workforce and U.S. economy. From this information, applications are developed to facilitate the development and maintenance of a skilled workforce.
It has detailed descriptions of the world of work for use by job seekers, workforce development and HR professionals, students, developers, researchers, and more!

Out of wide variety of Onet data. We have choosen Jobzone data for our classification experiments.

#### Job Zone
A Job Zone is a group of occupations that are similar in:

how much education people need to do the work,
how much related experience people need to do the work, and
how much on-the-job training people need to do the work.
The five Job Zones are:

- Job Zone 1 external site - occupations that need little or no preparation
- Job Zone 2 external site - occupations that need some preparation
- Job Zone 3 external site - occupations that need medium preparation
- Job Zone 4 external site - occupations that need considerable preparation
- Job Zone 5 external site - occupations that need extensive preparation

[Sample data can be viewed from this link](https://www.onetonline.org/find/zone?z=0)

#### Occupation  Data

The Occupation data is having the occupation title and duties description of the corresponding occupation.

Structure and Description of data as follows:

 | Column         | Type	| Column Content |
 |----------------|---------|--------------|
 | O*NET-SOC Code |	Character(10) | O*NET-SOC Code |
| Title	         |Character Varying(150)	|O*NET-SOC Title|
| Description    |Character Varying(1000)|O*NET-SOC Description|

Entire Occupations are categorised into 23 major groups called Domains
- Domains are as follows:
  - Management Occupations
  - Business and Financial Operations Occupations
  - Computer and Mathematical Occupations
  - Architecture and Engineering Occupations
  - Life, Physical, and Social Science Occupations
  - Community and Social Service Occupations
  - Legal Occupations
  - Educational Instruction and Library Occupations
  - Arts, Design, Entertainment, Sports, and Media Occupations
  - Healthcare Practitioners and Technical Occupations
  - Healthcare Support Occupations
  - Protective Service Occupations
  - Food Preparation and Serving Related Occupations
  - Building and Grounds Cleaning and Maintenance Occupations
  - Personal Care and Service Occupations
  - Sales and Related Occupations
  - Office and Administrative Support Occupations
  - Farming, Fishing, and Forestry Occupations
  - Construction and Extraction Occupations
  - Installation, Maintenance, and Repair Occupations
  - Production Occupations
  - Transportation and Material Moving Occupations

Totally there are **923** Occupations with the domain data is considered and preprocessed.

Preprocessed data and code can be found in the directory:

    --- Project
        |--- Data
               |--- Processed_data.csv
        |--- src
              |--- data_preprocessing.py

Due to imbalance in Job Zone data the data is up sampled and the counts for each class is as follows:

____ To Do ___ get  counts of job zone

    --- Project
        |--- src
              |--- data_balancing.py


## PreTrained Language Embeddings

There are different attention based multi-language models are present out of that 5 models are chosen for experiment

### Multilingual Bert

BERT is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then use that model for downstream NLP tasks that we care about (like question answering). BERT outperforms previous methods because it is the first unsupervised, deeply bidirectional system for pre-training NLP.

Unsupervised means that BERT was trained using only a plain text corpus, which is important because an enormous amount of plain text data is publicly available on the web in many languages.

Bert has basic building blocks of [transformer blocks](https://arxiv.org/pdf/1706.03762.pdf) used in encoder and decoder architecture.
<p align="center"> 
  <img width="700" src="project/referrences/BertBuilding block.png">
</p>

For tokenization, using a 110k shared WordPiece vocabulary. The word counts are weighted the same way as the data, so low-resource languages are upweighted by some factor. 
Because Chinese (and Japanese Kanji and Korean Hanja) does not have whitespace characters, added spaces around every character in the CJK Unicode range before applying WordPiece. This means that Chinese is effectively character-tokenized. 

For all other languages, apply the same recipe as English: (a) lower casing+accent removal, (b) punctuation splitting, (c) whitespace tokenization. We understand that accent markers have substantial meaning in some languages, but felt that the benefits of reducing the effective vocabulary make up for this.

Multilingual Bert will support 108 languages trained from MNLI dataset, wikipedia, language translation data etc.. 
**MBert model consists of 12-layer, 768-hidden, 12-heads, 110M parameters**

### Distill Multilingual Bert


## Referrences
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
*   Pires, Telmo and Schlinger, Eva and Garrette, Dan. [How multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502)
*   [Multilingual Bert Details](https://github.com/google-research/bert/blob/master/multilingual.md)
*   Jaderberg, M., Simonyan, K. and Zisserman, A., 2015. Spatial transformer networks. In Advances in neural information processing systems (pp. 2017-2025).
*   Spatial Transformer Networks by Kushagra Bhatnagar. https://link.medium.com/0b2OrmqVO5
*   [visualizing word Embeddings in lower dimension](https://medium.com/analytics-vidhya/word-embedding-using-python-63770334841)
*   [TensorFlow implementation training neural networks](https://www.tensorflow.org/guide/keras/train_and_evaluate)
*   [About O*net Ocuupation Data](https://www.onetcenter.org/taxonomy.html#latest)


