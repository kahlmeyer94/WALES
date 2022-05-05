import utils
import wales

graph, idx2word, word2idx = utils.load_graph()

emb_dict = utils.load_embedding('glove_50.p')

challenges = utils.load_challenges()

metric = wales.WALES(graph, word2idx, idx2word, emb_dict, gamma=1.0)
score = metric.evaluate(challenges, verbose=True)
print(f'The WALES score is {score}')