# Standard
import math
import matplotlib.pyplot  as plt
import numpy as np
import os
import shutil
from tqdm import tqdm
from collections import Counter

# Internet
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import cv2

# Others
from scipy import spatial
import networkx as nx
import language


'''
Webscraping
'''

def process_title(title):
    return [w.replace('(', '').replace(')','').lower() for w in title.split('_')]

def scrape_keyword(headers, img_shape=None, max_imgs=None, keyword='Special:Random', allowed_words=language.corpus, max_words=0, get_links=True, max_links=0, **kwargs):
    '''
    Scrapes a Wikipedia Article for links and images

    @Params:
        headers...          Dict, For wikipedia scraping, agent must identify itself 
                            (see https://meta.wikimedia.org/wiki/User-Agent_policy)
        img_shape...        Tuple, height and width of images; if None, no images are scraped
        max_imgs...         Int, if set will only take first max_imgs images (speed up!)
        keyword...          String, article to scrape (Default random article)
        allowed_words...    Set of words that mark allowed links
        max_words...        Maximum number of words (in articles paragraphs) to extract
        get_links...        If True, will scrape all outgoing links
        max_links...        If set > 0, will scrape only the most frequent occuring links
    @Returns:
        dictionary with
            'title'...          title of article, 
            'next_links'...     next_links, 
            'imgs'...           list of images, 
            'words'...          list of words,
            'word_counts'...    list of counts of words
    '''
    check_corpus = allowed_words
    
    next_links = []
    imgs = []
    words = []
    heading = keyword

    if get_links or (img_shape is not None) or max_words>0 or keyword=='Special:Random':
        url = f'https://en.wikipedia.org/wiki/{keyword}' 
        r = requests.get(url, headers=headers)
        
        if r.status_code == 200:
            content = r.content
            soup = BeautifulSoup(content, 'html.parser')
            
            # get name of article
            heading = soup.find(id='firstHeading').text.lower().replace(' ','_')
            if keyword=='Special:Random':
                keyword = heading

            # get links
            if get_links:
                next_links = []
                allLinks = soup.find(id="bodyContent").find_all("a")
                for link in allLinks:
                    try:
                        idx = link['href'].find("/wiki/")
                        if idx != -1:
                            link_title = link['href'][idx+6:] # get link
                            words = process_title(link_title)
                            if not '(disambiguation)' in link_title:
                                valid_title = True
                                for word in words:
                                    valid_title = valid_title and (word in check_corpus)
                                if valid_title:        
                                    next_links.append(link_title)
                    except KeyError:
                        pass 

                if max_links>0:
                    next_links = [x[0] for x in Counter(next_links).most_common(max_links)]
                next_links = list(set(next_links))
            
            # get images
            if img_shape is not None:
                allimgs = soup.find(id="mw-content-text").find_all("img")
                if max_imgs is None:
                    max_imgs = len(allimgs)
                count = 0
                for img in allimgs:
                    img_url = img["src"]
                    if 'commons/thumb/' in img_url:
                        if not img_url.startswith("http"):
                            img_url = "https:" + img_url
                        img = url_to_img(img_url, headers, img_shape)
                        imgs.append(img)
                        count += 1
                    if count >= max_imgs:
                        break
            
            # get paragraphs
            word_counts = []
            words = []
            if max_words > 0:
                main_text = soup.find(id='mw-content-text')
                paragraphs = main_text.findAll('p')
                
                words = {}
                total_text = []
                for p in paragraphs:
                    nouns = [token.lemma_ for token in language.nlp(p.text) if token.pos_ == 'NOUN' and token.lemma_ in check_corpus]
                    for n in nouns:
                        if n in words:
                            words[n] += 1
                        else:
                            words[n] = 1
                    total_text += nouns
                count_words = sorted([(n,words[n]) for n in words], key=lambda x: -x[1])
                top_words = [w[0] for w in count_words[:max_words]]
                words = [w for w in total_text if w in top_words]
                            
            if len(imgs)>0:
                imgs = np.stack(imgs)
        else:
            return None
    return {'title': keyword, 'heading': heading, 'next_links' : next_links, 'imgs' : imgs, 'words' : words}

def url_to_img(url, headers, img_shape):
    '''
    Returns image from URL

    @Params:
        url...          String, adress of image
        headers...      Dict, For wikipedia scraping, agent must identify itself 
                        (see https://meta.wikimedia.org/wiki/User-Agent_policy)
        img_shape...    Tuple, height and width of images

    @Returns:
        Numpy array HxWx3
    '''

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        img = np.array(Image.open(io.BytesIO(response.content)))
        if len(img.shape)==2:
            # grayscale
            img = np.transpose(np.stack([img,img,img]), (1,2,0))
        if img.shape[2]>3:
            # alpha channel
            img = img[:,:,:3]
            
        if img.shape[2]!=3:
            img = np.mean(img, axis=2).astype(np.uint8)
            img = np.transpose(np.stack([img,img,img]), (1,2,0))
        img = cv2.resize(img, (img_shape[1], img_shape[0]))
        
    else:
        img = None
    return img


'''
Distance Functions
'''

def cosine_dist(v1, v2):
    d = spatial.distance.cosine(v1, v2)
    if np.isnan(d):
        return np.inf
    else:
        return d

def euclid_dist(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

'''
Dataset
'''

def select_random_challenge(word_corpus, min_links=1):
    found = False
    while not found:
        start, target = np.random.choice(list(word_corpus), 2)

        headers = {
                'User-Agent': 'WikiRaceBot/Collector',
                'From': 'paul.kahlmeyer@uni-jena.de'
        } 
        resp1 = scrape_keyword(headers, keyword=start)
        resp2 = scrape_keyword(headers, keyword=target)
        if resp1 is not None and resp2 is not None:
            node1, links1 = resp1['heading'], resp1['next_links']
            node2, links2 = resp2['heading'], resp2['next_links']
            start_node = node1.lower()
            target_node = node2.lower()
            found = start_node in word_corpus and target_node in word_corpus
            found = found and len(links1)>=min_links and len(links2)>=min_links
    return start_node, target_node

def grow_wiki_graph(start_words, min_links, min_nodes, headers, max_leaf_nodes, scrape_params = {}, **kwargs):
    params = scrape_params
    params['headers'] = headers
    resp_dict = {}

    # 1 Initial graph
    dg = nx.DiGraph()
    for word in start_words:
        dg.add_node(word.lower())
    
    # 2 Grow
    largest_comp = max(nx.strongly_connected_components(dg), key=len)
    running = len(largest_comp) < min_nodes
    while running:
        
        # 2.1 Select leaf node to grow
        leaf_nodes = [x for x in dg if dg.out_degree(x)==0]
        
        if len(leaf_nodes)>max_leaf_nodes:
            diff = len(leaf_nodes)-max_leaf_nodes
            # prune graph
            del_nodes = np.random.choice(leaf_nodes, size=diff, replace=False)
            dg.remove_nodes_from(del_nodes)
            leaf_nodes = [x for x in leaf_nodes if x not in del_nodes]
        
        
        # weight proportional to number of 'siblings' with 
        probs = []
        for n in leaf_nodes:
            
            max_degree = 0
            for p in dg.predecessors(n):
                max_degree = max(max_degree, max([dg.out_degree(c) for c in dg.successors(p)]))
            probs.append(max_degree+1)
        probs = np.array(probs)
        probs = probs/np.sum(probs)
        
        if len(leaf_nodes)==0:
            print('No leaf nodes!')
            running = False
        else:
            grow_node = np.random.choice(leaf_nodes, p=probs)

            # 2.2 Grow node
            params['keyword'] = grow_node
            resp = scrape_keyword(**params)
            if resp is not None:
                resp_dict[grow_node] = resp
                next_links = resp['next_links']
                # 2.3 Add to graph
                if len(next_links)>min_links or grow_node in start_words:
                    print(f'Adding {grow_node}')
                    for word in next_links:
                        label = word.lower()
                        if label != grow_node and label in language.corpus:
                            dg.add_node(label)
                            dg.add_edge(grow_node, label)
                else:
                    del resp_dict[grow_node]
                    dg.remove_node(grow_node)
            
            largest_comp = max(nx.strongly_connected_components(dg), key=len)
            running = len(largest_comp) < min_nodes
            print(f'Largest Component {len(largest_comp)}\tNumber of nodes: {len(dg.nodes())}')
    
    ret_graph = dg.subgraph(largest_comp)
    ret_dict = {k : resp_dict[k] for k in resp_dict if k in ret_graph.nodes()}
    return ret_graph, ret_dict

def create_ds(graph, n_data, min_path_length=-math.inf, max_path_length=math.inf):
    # Dataset
    assert nx.is_strongly_connected(graph)
    ds = []
    for _ in range(n_data):
        found = False
        while not found:
            start_node, target_node = np.random.choice(list(graph.nodes), size=2, replace=False)
            path_length = len(nx.shortest_path(graph, source=start_node, target=target_node))
            found = path_length>=min_path_length and path_length<=max_path_length
        ds.append((start_node, target_node, path_length))
        
    return ds