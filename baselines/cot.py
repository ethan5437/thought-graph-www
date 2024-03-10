from prompts import cot
from models import gpt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

glove_path = '/Users/kylecox/Downloads/glove/glove.6B.300d.txt'


def get_cosine_similarity(sentence1, sentence2):
    glove_path = '/Users/kylecox/Downloads/glove/glove.6B.300d.txt'
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector

    tokens1 = sentence1.split()
    tokens2 = sentence2.split()
    embeddings1 = np.mean([embeddings[token] for token in tokens1 if token in embeddings], axis=0)
    embeddings2 = np.mean([embeddings[token] for token in tokens2 if token in embeddings], axis=0)
    similarity = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))
    return similarity[0][0]

def prompt_wrap(x):
    return cot.prompt.format(x=x)

def parse_answer(answer):
    # Answer should end with "So, the answer is <answer>."
    # Return <answer>
    try:
        ans = answer.split("the answer is ")[1].split(".")[0]
        # remove quotes
        ans = ans.replace('"', '')
        return ans
    except:
        return answer

def test_example(x, y):
    prompt = prompt_wrap(x)
    answer = parse_answer(gpt(prompt)[0])
    print('Exact Match: ', answer == y)
    return answer

def test_all():
    test = pd.read_csv('data/test.csv')[['Genes', 'Term_Description']]
    test = test.rename(columns={'Genes': 'x', 'Term_Description': 'y'})
    test = test.to_dict('records')
    for i in test:
        ans = test_example(i['x'], i['y'])
        print('Gold: ', i['y'])
        print('Predicted: ', ans)
        print('Score: ', round(get_cosine_similarity(i['y'], ans), 3))
        print('---------------------------------')


if __name__ == '__main__':
    test_all()


