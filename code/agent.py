import utils
import networkx as nx
import numpy as np
import time
import language


'''
Wiki Race
'''

def wikiRace(agent, graph, start_node, target_node, resp_dict, max_it=None):
    '''
    Starts a WikiRace on a given graph.
    This has the advantage, that you know the optimal path beforehand.
    
    @Params:
        agent...        Instance of Agent class
        graph...        Networkx graph of names of wiki articles
        start_node...   Start article
        target_node...  Target article
        resp_dict...    dictionary, that holds responses for every node in the graph
        max_it...       maximum iterations until a wikiRace is considered a fail

    @Returns:
        path... List of articles visited
        converged... boolean, indicating if the agent found target within max_it iterations
    '''

    if max_it is None:
        max_it = 2*len(graph.nodes)
    
    current_node = start_node
    target_resp = resp_dict[target_node]

    path = [current_node]
    count = 0
    while current_node!=target_node and count<max_it:
        successors = sorted(list(graph.successors(current_node)))
        current_resp = resp_dict[current_node]
        successor_resps = [resp_dict[s] for s in successors]
        current_node = agent.select(current_resp, successor_resps, target_resp)
        path.append(current_node)
        count += 1
    
    return path, count<max_it


def wikiRaceReal(agent, start_node, target_node, scrape_params={}, max_it=100, print_progress=True, **kwargs):
    '''
    Starts a real WikiRace on the english Wikipedia.

    @Params:
        agent...            Instance of Agent class
        start_node...       Start article
        target_node...      Target article
        scrape_params...    parameters for utils.scrape
        max_it...           Maximum number of iterations before considered fail
        print_progress...   Boolean, if True prints status messages

    @Returns:
        path...     List of visited nodes
        success...  Boolean indicating the convergence within max_it iterations
    '''
    
    params = scrape_params
    headers = {
            'User-Agent': 'WikiRaceBot/Racer',
            'From': 'paul.kahlmeyer@uni-jena.de'
    }
    params['headers'] = headers
    
    resp_dict = {}

    if print_progress:
        print(f'Navigating from {start_node} to {target_node}')
        print(start_node)
    # get target response
    params['keyword'] = target_node
    target_resp = utils.scrape_keyword(**params)
    resp_dict[target_node] = target_resp

    if target_resp is not None:
        # start iteration
        current_node = start_node
        path = [current_node]
        count = 0
        success = True
        while current_node.lower() != target_node.lower() and count<max_it and success:
            if current_node in resp_dict:
                current_resp = resp_dict[current_node]
            else:
                params['keyword'] = current_node
                params['get_links'] = True
                current_resp = utils.scrape_keyword(**params)
                if current_resp is not None:
                    resp_dict[current_node] = current_resp
            deadend = False
            if current_resp is not None:
                # collect response for successors

                successor_resps = []
                for s in current_resp['next_links']:
                    print(f'Scraping Sucessor: {s}')
                    if s in resp_dict:
                        successor_resps.append(resp_dict[s])
                    else:
                        params['keyword'] = s
                        params['get_links'] = False
                        resp = utils.scrape_keyword(**params)
                        if resp is not None:
                            successor_resps.append(resp)
                
                deadend = len(successor_resps)==0
                if not deadend:
                    current_node = agent.select(current_resp, successor_resps, target_resp)
                    count += 1
                    path.append(current_node)
            else:
                count = max_it
                print(f'could not resolve url for {current_node}!')

            if current_node is None or deadend:
                # Agent did not succeed
                success = False
            if print_progress:
                print(current_node)

        if count == max_it:
            success = False


        return path, success
    else:
        print('Could not reach target page!')
        return None

'''
Agents
'''

class Agent():
    def __init__(self):
        pass
    
    def select(self, current_resp, successor_resps, target_resp):
        # resp is a dictionary with keys
        pass
    
class RandomAgent(Agent):
    
    def select(self, current_resp, successor_resps, target_resp):
        return np.random.choice([resp['title'] for resp in successor_resps])
    
class HumanAgent(Agent):
    def select(self, current_resp, successor_resps, target_resp):
        target_node = target_resp['title']
        print(f"Current node: {current_resp['title']}\nChoose your next node (Target: {target_node})")
        for i,resp in enumerate(successor_resps):
            print(f"({i})...\t{resp['title']}")
        i = -1
        min_idx = 0
        max_idx = len(successor_resps)-1
        while i < min_idx or i > max_idx:
            i = int(input(''))
        
        selected_node = successor_resps[i]['title']
        print(f'Moving to {selected_node}')
        return selected_node
       
class GraphGreedyAgent(Agent):
    '''
    Greedy Routing using an internal graph to select one of the reachable leaf nodes.
    '''

    def __init__(self, perception, dist_func):
        self.memory_map = nx.DiGraph() # graph of already visited nodes + their neighbours
        self.track_path = [] # in case we want to navigate to a leaf node: for speed up
        self.perception = perception # responses -> vectors [e.g. perception.lda_perception(x, lda_model)]
        self.dist_func = dist_func # vector,vector -> float [e.g. utils.euclid_dist]
        self.perception_dict = {} # dictionary to store all perceptions

    def reset(self):
        self.memory_map = nx.DiGraph()
        self.track_path = []
        self.perception_dict = {}
           
    def select(self, current_resp, successor_resps, target_resp):
        
        target_node = target_resp['title']
        current_node = current_resp['title']
        successor_nodes = [resp['title'] for resp in successor_resps]

        for s in successor_nodes:
            if s not in self.memory_map.nodes:
                self.memory_map.add_node(s)
            if (current_node, s) not in self.memory_map.edges:
                self.memory_map.add_edge(current_node, s)
            if s.lower()==target_node.lower():
                return s

        if len(self.track_path) > 0:
            next_node = self.track_path.pop(0)
            assert next_node in successor_nodes
            return next_node
        
        else:
            # Add to internal graph
            if not current_node in self.memory_map:
                self.memory_map.add_node(current_node)
            new_resps = []
            for i,suc_node in enumerate(successor_nodes):
                new_resps.append(successor_resps[i])
                
            new_percs = self.perception(new_resps)
            for resp, perc in zip(new_resps, new_percs):
                self.perception_dict[resp['title']] = perc 

            desc =  nx.descendants(self.memory_map, current_node)
            if len(desc)==0:
                return None

            elif target_node in desc:
                # means, target must be reachable from current node and be in memory_map
                tmp = nx.shortest_path(self.memory_map, source=current_node, target=target_node)
                next_node = tmp[1]
                self.track_path = tmp[2:]
            
            else:
                # Which successors have the shortest path to a leaf node?

                leaf_nodes = [n for n in desc if self.memory_map.out_degree(n)==0]
                
                if len(leaf_nodes)==0:
                    # we are in some kind of loop
                    next_node = np.random.choice(successor_nodes)
                else:
                    sp = nx.shortest_path(self.memory_map, source=current_node)
                    min_paths = [sp[n] for n in leaf_nodes]
                    min_path_len = [len(p) for p in min_paths]

                    percepts = [self.perception_dict[n] for n in leaf_nodes]
                    if target_node not in self.perception_dict:
                        self.perception_dict[target_node] = self.perception([target_resp])
                    target_percept = self.perception_dict[target_node]

                    dists = [self.dist_func(p ,target_percept) for p in percepts]

                    values = [d for d,l in zip(dists,min_path_len)]
                    idx = np.argmin(values)
                    min_path = min_paths[idx]
                    next_node = min_path[1]
                    self.track_path = min_path[2:]
                    
            return next_node



    
    