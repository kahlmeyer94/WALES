import utils
from tqdm import tqdm
import numpy as np
import spacy
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import gutenberg

import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath
from gensim.models import CoherenceModel, LdaModel

# In case not already downloaded, uncomment:
#nltk.download('wordnet')
#nltk.download('gutenberg')

# Parsing paragraphs
nlp = spacy.load('en_core_web_md')
nlp_corpus = set(nlp.vocab.strings)
nlp_check = lambda x: x in nlp_corpus and np.any(nlp(x).vector != 0)


# Reasonable words
corpus = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
common_dictionary = corpora.Dictionary([list(corpus)])

def train_lda_model(documents,  lda_params={}, model_name=None, coherence=True):
    '''
    Trains a LDA Model.

    @Params:
        documents...    List of lists of words
        lda_params...   Parameters to be passed to gensims ldamodel 
                        (see https://radimrehurek.com/gensim/models/ldamodel.html)
        save_path...    string, path to save lda model
        coherence...    if True, will calculate the coherence on the trainingdata
    '''
    corpus = [common_dictionary.doc2bow(text) for text in documents] 
    lda_params['corpus'] = corpus
    lda_params['id2word'] = common_dictionary
    lda_model = LdaModel(**lda_params)

    if coherence:
        coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, dictionary=common_dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f'Coherence: {coherence_lda}')

    if model_name is not None:
        temp_file = datapath(model_name)
        lda_model.save(temp_file)
        pass
    return lda_model

def get_gutenberg_documents(batchsize=5000, files=None):
    '''
    Creates a collection of documents out of the gutenberg collection of nltk.

    @Params:
        batchsize...    Number of words considered as batch (in order for nlp to parse)
        files...        names of files to use. If None, uses all files

    @Returns:
        List of documents. Each Document is itself a list of words.
    '''
    #get gutenberg corpus
    ids = gutenberg.fileids()
    articles = []

    if files is None:
        files = ids
    batchsize = 5000
    for g_id in ids:
        if g_id in files:
            print(f'File: {g_id}')
            words = gutenberg.words(g_id)
            n_batches = int(len(words)/batchsize)
            print(f'Number of batches: {n_batches}')
            for i in range(n_batches):
                part = words[i*batchsize:(i+1)*batchsize]
                part = ' '.join(x for x in part)
                tmp = nlp(str(part))
                articles.append([token.lemma_ for token in tmp if token.pos_ == 'NOUN' and token.lemma_ in corpus])
    return articles

def get_wiki_documents(N_articles, keywords=[], topwords=20):
    '''
    Creates a collection of documents scraped from wikipedia.

    @Params:
        N_articles...   number of documents to collect
        keywords...     articles to cover regardless
        topwords...     consider only top frequent words

    @Returns:
        List of documents. Each Document is itself a list of words.
    '''
    headers = {
        'User-Agent': 'WikiRaceBot/LDAAgent',
        'From': 'paul.kahlmeyer@uni-jena.de'
    }
    articles = [] 
    # collect keywords
    for kw in tqdm(keywords):
        resp = utils.scrape_keyword(headers=headers, keyword=kw, max_words=topwords)      
        if resp is not None:
            articles.append(resp['words'])
    
    # fill up with random articles
    random_articles = max(0, N_articles-len(keywords))
    for _ in tqdm(range(random_articles)):
        resp = utils.scrape_keyword(headers=headers, max_words=topwords)      
        if resp is not None:
            articles.append(resp['words'])
    return articles