from sentence_transformers import SentenceTransformer
import numpy as np
sentences = ["This is an example sentence", "spit"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

set1 = ["rise up, wake up", "be skilled at, be adept at, well versed in; be enough?"]
set2 = ["awaken", "able"]

PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/embeddings/"
emb_set_1 = model.encode(set1)
emb_set_2 = model.encode(set2)
# print(emb_set_1.shape)

np.save(f'{PATH}1.npy', emb_set_1)
np.save(f'{PATH}2.npy', emb_set_2)

es1 = np.load(f'{PATH}1.npy')
es2 = np.load(f'{PATH}2.npy')

for e1 in es1:
    for e2 in es2:
        print(np.dot(e1, e2))

