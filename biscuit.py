import os
import numpy as np
import pandas as pd

DATA_PATH = "/home/aaditd/4_Subword/miniproject-4/subword-miniproject-4/datase/"

languages = ["huishu", "kachai", "tusom"]
language = languages[0]

CANDIDATE_PATH = f"{DATA_PATH}candidates/ukhrul-{language}_inputs.tsv"
INPUT_PATH = f"{DATA_PATH}inputs/ukhrul-{language}_inputs.tsv"


def read_df(filename):
    df = pd.read_csv(filename, sep="\t", names=['word', 'gloss'], header=None)
    return df["word"], df["gloss"]


inputs, i_glosses = read_df(INPUT_PATH)
candidates, c_glosses = read_df(CANDIDATE_PATH)

print(inputs[:5])
print(i_glosses[:5])

print()

print(candidates[:5])
print(c_glosses[:5])

def phonological_similarity(u_word, l_word):
    a = 0

    return 77


def semantic_similarity(u_word, l_word):
    a = 0

    return 666


def evaluate(gold, inputs, preds):
  total = []
  for idx, inp in enumerate(inputs):
    if gold["\t".join(inp)] in preds[idx][:5]:
      total.append(1.0 / (preds[idx].index(gold["\t".join(inp)]) + 1))
  
  return sum(total) / len(total) 