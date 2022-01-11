import networkx as nx
import numpy as np
from tqdm import tqdm


import utils

class WALES():

    def __init__(self, wiki_graph, word2idx, idx2word, emb_dict, gamma=1.0, debug=False, **params):
        '''
        @Params:
            wiki_graph...   Networkx directed graph
            word2idx...     Dictionary, key=article name, value=node idx
            idx2word...     Dictinoary, key=node idx, value=article name
            emb_dict...     Dictionary, key=word, value=embedding vector
            gamma...        parameter for search
        '''
        self.wiki_graph = wiki_graph
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.d = len(emb_dict[list(emb_dict.keys())[0]])
        self.gamma = gamma
        self.internal_graph = nx.DiGraph()

        self.idx2emb = {}
        for i in self.wiki_graph.nodes():
            w = self.idx2word[i]

            parts = utils.process_title(w)
            vecs = []
            for p in parts:
                if p in emb_dict:
                    vecs.append(emb_dict[p])
                else:
                    vecs.append(np.zeros(self.d))
            vecs = np.stack(vecs)
            self.idx2emb[i] = np.mean(vecs, axis=0)
            
    def route(self, start, target, max_it=100):
        '''
        @Params:
            start...        Start article
            target...       Target article
            max_it...       Abortion after this many iterations
        '''
        start_idx = self.word2idx[start]
        target_idx = self.word2idx[target]
        v_target = self.idx2emb[target_idx]

        current_idx = start_idx
        p = []
        visited_nodes = set()

        p.append(self.idx2word[current_idx])
        visited_nodes.add(current_idx)
        random_counts = 0
        while current_idx != target_idx and len(p)<max_it:
            if current_idx not in self.internal_graph.nodes():
                self.internal_graph.add_node(current_idx)
            neighbors = [n for n in nx.neighbors(self.wiki_graph, current_idx)]
            new_neighbors = []
            for n in neighbors:
                if n not in visited_nodes:
                    self.internal_graph.add_node(n)
                    new_neighbors.append(n)
                self.internal_graph.add_edge(current_idx, n)
            

            if self.gamma==1.0 and len(new_neighbors)>0:
                # select greedy neighbor (=speed up)
                embs = [self.idx2emb[n] for n in new_neighbors]
                if len(embs)>0:
                    dists = utils.cosine_dist_parallel(np.stack(embs), v_target)
                    current_idx = new_neighbors[np.argmin(dists)]
                p.append(self.idx2word[current_idx])
                visited_nodes.add(current_idx)
            else:
                reachable_leaves = [n for n in nx.descendants(self.internal_graph, current_idx) if self.internal_graph.out_degree(n)==0]
                shortest_paths = [nx.shortest_path(self.internal_graph, source=current_idx, target=n) for n in reachable_leaves]
                V = np.stack([self.idx2emb[n] for n in reachable_leaves])
                dists = utils.cosine_dist_parallel(V, v_target)
                dists = [dists[i]+self.gamma*len(shortest_paths[i]) for i in range(len(reachable_leaves))]
                min_idx = np.argmin(dists)
                current_idx = reachable_leaves[min_idx]
                p = p+[self.idx2word[w] for w in shortest_paths[min_idx][1:]]
        return p, p[-1]==target

    def evaluate(self, challenges, verbose=False):
        '''
        Evaluates WALES metric.

        @Params: 
            challenges...   list of start,target tuples (words)

        @Returns:
            WALES score (float),
            Percent of failed runs (float)
        '''


        results = []

        if verbose:
            pbar = tqdm(challenges)
        else:
            pbar = challenges
        for s_node, t_node in pbar:
            p, converged = self.route(s_node, t_node)
            self.internal_graph = nx.DiGraph()

            if converged:
                opt_path = nx.shortest_path_length(self.wiki_graph, self.word2idx[s_node], self.word2idx[t_node])+1
                results.append(opt_path/len(p))
            else:
                results.append(0)
        results = np.array(results)
        return np.mean(results)