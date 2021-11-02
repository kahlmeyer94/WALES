import spacy
import numpy as np
import utils
import language
import perception



def word2vec_perception(responses):
    titles = [utils.process_title(resp['title']) for resp in responses]
    vectors = []
    for words in titles:
        v = []
        for w in words: 
            #assert language.nlp_check(w)
            v.append(language.nlp(w).vector)
        vectors.append(np.mean(np.stack(v), axis=0))
    return np.stack(vectors)

def lda_perception(responses, lda_model):
    tmp = [language.common_dictionary.doc2bow(resp['words']) for resp in responses] 
    vectors = []
    for i in range(len(responses)):
        vec = np.array([x[1] for x in lda_model[tmp[i]]])
        vectors.append(vec)
    return np.stack(vectors)