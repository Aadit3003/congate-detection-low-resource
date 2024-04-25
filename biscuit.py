import os
import numpy as np
import pandas as pd
from utils import file_write_strings

DATA_PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/dataset/"
# "/home/aaditd/4_Subword/P/subword-miniproject-4/dataset/candidates/huishu_candidates.tsv"

languages = ["huishu", "kachai", "tusom"]
language = languages[0]

OUTPUT_PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/outputs/"

CANDIDATE_PATH = f"{DATA_PATH}candidates/{language}_candidates.tsv"
INPUT_PATH = f"{DATA_PATH}inputs/ukhrul-{language}_inputs.tsv"
GOLD_PATH = f"{DATA_PATH}/ukhrul-huishu_gold.tsv"

def make_dictionary(words, glosses):
    gloss_dic = {}
    for w, g in zip(words, glosses):
        if w not in list(gloss_dic.keys()):
            gloss_dic[w] = [g]
        else:
            gloss_dic[w].append(g)

    return gloss_dic

def read_df(filename):
    df = pd.read_csv(filename, sep="\t", names=['word', 'gloss'], header=None)

    words, glosses = list(df["word"]), list(df["gloss"])
    # gloss_dic = make_dictionary(words, glosses)


    return list(zip(words, glosses))

def read_gold_df(filename):
    df = pd.read_csv(filename, sep="\t", names=['u_word', 'u_gloss', 'cognate', 'cognate_gloss'], header=None)

    u_words = list(df["u_word"])
    u_gloss = list(df["u_gloss"])
    gold_cognates = list(df["cognate"])
    gold_gloss = list(df['cognate_gloss'])

    gold_dic = {'\t'.join([uw, ug]):'\t'.join([gc, gg])for uw, ug, gc, gg in zip(u_words, u_gloss, gold_cognates, gold_gloss)}
    return gold_dic

def write_output_df(preds):
   
    output_filename = f"{OUTPUT_PATH}ukhrul-{language}_candidates.tsv"
    list_version_pred = []


    for lis in preds:
        lis_version_per_word = []

        for pair in lis:
            lis_version_per_word.append(pair[0])
            lis_version_per_word.append(pair[1])

        list_version_pred.append(lis_version_per_word)

    keys = ["c1", "g1", "c2", "g2", "c3", "g3", "c4", "g4", "c5", "g5"]

    write_dic = {}
    for count, k in enumerate(keys):
        write_dic[k] = [l[count] for l in list_version_pred]

      
    df = pd.DataFrame(write_dic)
    df.to_csv(output_filename, sep="\t", header=False, index=False)
    print("Output File Written!")
   



INP_GLOSS_PAIRS = read_df(INPUT_PATH)
CAND_GLOSS_PAIRS = read_df(CANDIDATE_PATH)

# make_dictionary(inputs, i_glosses)
# make_dictionary(candidates, c_glosses)

print(f"Unique Ukhrul Input words: {len(INP_GLOSS_PAIRS)}")
print(f"Unique {language} Candidate words: {len(CAND_GLOSS_PAIRS)}")

GOLD_DIC = read_gold_df(GOLD_PATH)
print()
assert len(GOLD_DIC) == len(INP_GLOSS_PAIRS), f"Gold {len(GOLD_DIC)} and input file {len(INP_GLOSS_PAIRS)} have different number of entries!"

# print(INP_GLOSS_DIC["ka	climb, ascend, up; do in upward path (postp.)"])

def phonological_similarity(u_word, other_word):
    a = 0

    return 77


def semantic_similarity(u_word, u_gloss, other_word, other_gloss):
    global INP_GLOSS_DIC
    global CAND_GLOSS_DIC


    return int(u_gloss == other_gloss)

    return len(ig.intersection(cg))


def scoring_function(u_word, u_gloss, other_word, other_gloss):
   score = semantic_similarity(u_word, other_word)

   return score


def get_top_5_words(u_word, u_gloss):
    # candidates = list(CAND_GLOSS_DIC.keys())
    

    scores = []
    for c, cg in CAND_GLOSS_PAIRS:
        scores.append( (c, cg, scoring_function(u_word, u_gloss, c, cg)) )

    scores = sorted(scores, key=lambda x:x[1], reverse=True)

    top_5 = scores[:5]
    print(f"    Word: {u_word} | Preds: {top_5}")






#    top_5 = candidates[:5]

    return [(t, tg) for t, tg, s in top_5]
   

# preds = [get_top_5_words(u) for u in inputs]

def evaluate(inputs, preds):
  global GOLD_DIC
  """
  inputs: A list of Ukhrul words
    [
        u1,
        u2,
        ...
        u1000
    ]
  preds: A list of list of tuples containing the top-5 cognates and glosses for that word
    [ 
        [(c1, g1), (c2, g2), ..., (c5, g5)],
        [(c1, g1), (c2, g2), ..., (c5, g5)],
        ...
        [(c1, g1), (c2, g2), ..., (c5, g5)]
    ]

    inputs and preds are aligned!! (GOLD_DIC is not, so it's a dictionary!)
  """
  total = []

  for idx, inp in enumerate(inputs):
    gold_cognate = GOLD_DIC[inp]
    retrieved_cognates = ['\t'.join(p) for p in preds[idx]]
    if gold_cognate in retrieved_cognates:
      total.append(1.0 / (retrieved_cognates.index(gold_cognate) + 1))
  
  if len(total) == 0: return 0
  return sum(total) / len(total) 



test_inputs = INP_GLOSS_PAIRS[:5]

test_inputs = ['ʃa', 'ka']
# print(INP_GLOSS_PAIRS['ʃa'])

test_preds = [get_top_5_words(iw, ig) for iw, ig in test_inputs]
for t in test_preds[:5]:
    print(t)

print(evaluate(test_inputs, test_preds))
write_output_df(test_preds)


    
# df = pd.read_csv(f"{OUTPUT_PATH}ukhrul-{language}_candidates.tsv", sep="\t", header=None, names = ["c1", "g1", "c2", "g2", "c3", "g3", "c4", "g4", "c5", "g5"])
# print(list(df["c1"]))