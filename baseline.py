import numpy as np
import panphon



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
langB = 'huishu'


inputs = [l.strip().split('\t') for l in open(f'{DATA_PATH}inputs/{langA}-{langB}_inputs.tsv', 'r')]
candidates = [l.strip().split('\t') for l in open(f'{DATA_PATH}candidates/{langB}_candidates.tsv', 'r')]
if langB == 'huishu':
    gold = {'\t'.join(l.strip().split('\t')[:2]): '\t'.join(l.strip().split('\t')[2:])  for l in open(f'{DATA_PATH}{langA}-{langB}_gold.tsv', 'r')}


ft = panphon.FeatureTable()

longest_word = max([len(inp[0]) for inp in inputs] + [len(can[0]) for can in candidates])

input_features = np.array([pad_sequence(np.array(ft.word_to_vector_list(inp[0], numeric=True)) + 2, longest_word).reshape(-1) for inp in inputs])
print(len(input_features))
candidate_features = np.array([pad_sequence(np.array(ft.word_to_vector_list(can[0], numeric=True)) + 2, longest_word).reshape(-1)  for can in candidates])


preds = []

for idx in range(input_features.shape[0]):

  dists = np.dot(candidate_features, input_features[idx].T) / (np.linalg.norm(candidate_features, axis=1) * np.linalg.norm(input_features[idx]))
  defs = np.array([inputs[idx][1] == can[1] for can in candidates], dtype=float)
  best = np.argsort(dists + defs)[::-1]
  preds.append(["\t".join(candidates[i]) for i in best[:5]])


print(f"MRR is {round(mean_reciprocal_rank(gold, inputs, preds), 3)}")
save_preds(preds, langA, langB)
