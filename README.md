# Leveraging the Wikipedia Graph for Evaluating Word Embeddings

## Requirements
```
pip install -r requirements.txt
```

## Calculating WALES
A minimal code example for using the WALES metric can be found in `main.py`:
```
import utils
import wales
```

###  1. Load Wikipedia Subgraph
```
graph, idx2word, word2idx = utils.load_graph()
```
`graph` is a networkx graph, `idx2word` and `word2idx` are dictionaries to translate between node indices and article names.

### 2. Loading an embedding
```
emb_dict = utils.load_embedding('glove_50.p')
```
`emb_dict` has to be a dictionary (key=word, value=vector).

Note that for memory reasons, we provide only the pretrained GloVe (d=50) embedding used in the paper.
To reproduce all the results of the paper, download the embeddings and produce dictionaries for the nodes in the graph.

- [ELMo](https://allennlp.org/elmo)
- [fasttext](https://fasttext.cc/docs/en/english-vectors.html)
- [GloVe](https://nlp.stanford.edu/projects/glove/)
- [Word2Vec](https://code.google.com/archive/p/word2vec/)

### 3. Loading challenges
We evaluate WALES with a set of challenges.
To load the human benchmark dataset use
```
challenges = utils.load_challenges()
```
`challenges` has to be a list of word tuples, that occur in `word2idx`.

### 4. WALES
Simply instantiate the WALES class and run the `.evaluate` method.
```
metric = wales.WALES(graph, word2idx, idx2word, emb_dict, gamma=1.0)
score = metric.evaluate(challenges, verbose=True)
print(score)
```
In this example, the embedding got a WALES score of 0.54.
```
>>> 0.5437351135473592
```


