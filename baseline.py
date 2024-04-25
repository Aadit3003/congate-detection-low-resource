import numpy as np
import panphon
from panphon import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/dataset/"
OUTPUT_PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/outputs/"

def mean_reciprocal_rank(gold, inputs, preds):
  total = []
  for idx, inp in enumerate(inputs):
    if gold["\t".join(inp)] in preds[idx][:5]:
      total.append(1.0 / (preds[idx].index(gold["\t".join(inp)]) + 1))
    else:
      total.append(0.0)

  return sum(total) / len(total)

def pad_sequence(seq, length):
  return np.vstack([seq, np.ones((length - seq.shape[0], seq.shape[1]), dtype=float) * 1e-10])


def save_preds(preds, langA, langB):
  with open(f'{OUTPUT_PATH}{langA}-{langB}_out.tsv', 'w') as f:
    preds = ['\t'.join(p) for p in preds]
    f.write('\n'.join(preds))

langA = 'ukhrul'
# langB = 'huishu'
langB = 'kachai'

# Added 'ru' to the empty string in line 47, 184, and 211 of kachai_candidates.tsv


inputs = [l.strip().split('\t') for l in open(f'{DATA_PATH}inputs/{langA}-{langB}_inputs.tsv', 'r')]
print(len(inputs))
candidates = [l.strip().split('\t') for l in open(f'{DATA_PATH}candidates/{langB}_candidates.tsv', 'r')]



if langB == 'huishu':
    gold = {'\t'.join(l.strip().split('\t')[:2]): '\t'.join(l.strip().split('\t')[2:])  for l in open(f'{DATA_PATH}{langA}-{langB}_gold.tsv', 'r')}


ft = panphon.FeatureTable()

longest_word = max([len(inp[0]) for inp in inputs] + [len(can[0]) for can in candidates])

input_features = np.array([pad_sequence(np.array(ft.word_to_vector_list(inp[0], numeric=True)) + 2, longest_word).reshape(-1) for inp in inputs])
print(len(input_features))
candidate_features = np.array([pad_sequence(np.array(ft.word_to_vector_list(can[0], numeric=True)) + 2, longest_word).reshape(-1)  for can in candidates])

PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/embeddings/"
input_emb = np.load(f'{PATH}ukhrul.npy')
candidate_emb = np.load(f'{PATH}{langB}.npy')

corpus_1 = [w for w, g in inputs]
corpus_2 = [w for w, g in candidates]
vectorizer1 = TfidfVectorizer(analyzer= 'char', ngram_range = (1, 3), max_features = 100)
vectorizer2 = TfidfVectorizer(analyzer= 'char', ngram_range = (1, 3), max_features = 100)

corpus_3 = [g for w, g in inputs]
corpus_4 = [g for w, g in candidates]
vectorizer3 = TfidfVectorizer(max_features = 200)
vectorizer4 = TfidfVectorizer(max_features = 200)

X_inp = vectorizer1.fit_transform(corpus_1)
print(X_inp.shape)
X_cand = vectorizer1.fit_transform(corpus_2)
print(X_cand.shape)

X_ig = vectorizer1.fit_transform(corpus_3)
print(X_ig.shape)
X_cg = vectorizer1.fit_transform(corpus_4)
print(X_cg.shape)

d = distance.Distance()

# BEST COMBO - (1,3), max 100, 0.1*tfidf, 3*defs. 2 * dists
preds = []

for idx in range(input_features.shape[0]):

  dists = np.dot(candidate_features, input_features[idx].T) / (np.linalg.norm(candidate_features, axis=1) * np.linalg.norm(input_features[idx])) # Phonological Feats

  tf_idf_scores = cosine_similarity(X_inp[idx], X_cand)[0]

  gloss_scores = cosine_similarity(X_ig[idx], X_cg)[0]

#   tf_idf_scores = np.multiply(tf_idf_scores, 0.5*gloss_scores)

  embedding_scores = np.array([0.05*np.dot(input_emb[0], cand_emb) for cand_emb in candidate_emb])


#   fld = []
#   for can, _ in candidates:
#     distance = d.fast_levenshtein_distance_div_maxlen(inputs[idx][0], can)
#     if distance == 0: fld.append(1)
#     else: fld.append(0.5/distance)
  fld = [1 for _ in range(len(candidates))]


  defs = np.array([inputs[idx][1] == can[1] for can in candidates], dtype=float) # Semantic Exact Match

  best = np.multiply(2*dists + 3 *defs + 0.1 * tf_idf_scores + 0.1 * gloss_scores, fld)
  best = np.argsort(best)[::-1]
#   best = np.argsort(dists + defs)[::-1]
  preds.append(["\t".join(candidates[i]) for i in best[:5]])

if langB == "huishu":
    print(f"MRR is {round(mean_reciprocal_rank(gold, inputs, preds), 3)}")
save_preds(preds, langA, langB)
print("DONE!")
