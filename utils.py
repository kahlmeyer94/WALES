import numpy as np
from scipy import spatial
import networkx as nx
import os
import pickle

def process_title(title):
    return [w.replace('(', '').replace(')','').lower() for w in title.split('_')]

'''
Data
'''

data_path = 'data'
def load_graph():
    '''
    Loads the Wikipedia subgraph and the translation dictionaries.

    @Returns:
        graph...    networkx graph
        idx2word... dictionary (key=index, value=word)
        word2idx... dictionary (key=word, value=idx)
    '''

    graph = nx.read_gpickle(os.path.join(data_path, 'wales', 'wiki_graph.p'))
    with open(os.path.join(data_path, 'wales', 'idx2name.p'), 'rb') as handle:
        idx2word = pickle.load(handle)
    with open(os.path.join(data_path, 'wales', 'name2idx.p'), 'rb') as handle:
        word2idx = pickle.load(handle)  
    return graph, idx2word, word2idx

def load_challenges():
    '''
    Loads scraped Wikirace challenges.

    @Returns:
        challenges... list of word tuples
    '''

    challenges = []
    with open(os.path.join(data_path, 'wales', 'challenges.txt'), 'r') as inf:
        lines = inf.readlines()
    for line in lines:
        s, t = line.replace('\n', '').split('\t')
        challenges.append((s,t))
    return challenges

def load_embedding(emb_file):
    with open(os.path.join('data', 'embeddings', emb_file), 'rb') as handle:
        emb_dict = pickle.load(handle)
    return emb_dict

'''
Distance Functions
'''

def cosine_dist(v1, v2):
    '''
    Cosine distance between two vectors.

    @Params:
        v1... vector 1
        v2... vector 2
    @Returns:
        1-cossim
    '''

    d = spatial.distance.cosine(v1, v2)
    if np.isnan(d) or np.all(v1==0) or np.all(v2==0):
        return np.inf
    else:
        return d

def cosine_dist_parallel(V, v):
    '''
    Parallel cosine distance between batch of vectors and a vector.

    @Params:
        V...    batch of vectors (n x d)
        v...   vector (d)
    @Returns:
        1-cossim between each vector in batch and v
    '''

    mask = np.sum((V!=0), axis=1)==0
    tmp = V@v
    tmp[mask] = np.inf
    tmp[~mask] = 1 - tmp[~mask]/(np.linalg.norm(V[~mask,:], axis=1) * np.linalg.norm(v))
    return tmp