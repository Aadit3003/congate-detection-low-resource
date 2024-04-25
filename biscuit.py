import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from utils import file_write_strings
import editdistance
import argparse
import math
from panphon import distance
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
# nlp = spacy.load("en_core_web_lg")

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

def write_output_df(preds, OUTPUT_PATH, language):
   
    output_filename = f"{OUTPUT_PATH}ukhrul-{language}_out.tsv"
    list_version_pred = []


    for lis in preds:
        lis_version_per_word = []

        for pair in lis:
            lis_version_per_word.append(pair[0])
            lis_version_per_word.append(pair[1])

        list_version_pred.append(lis_version_per_word)

    keys = ["c1", "g1", "c2", "g2", "c3", "g3", "c4", "g4", "c5", "g5"]

    # assert len(list_version_pred) == len(INP_GLOSS_PAIRS)

    write_dic = {}
    for count, k in enumerate(keys):
        write_dic[k] = [l[count] for l in list_version_pred]

      
    df = pd.DataFrame(write_dic)
    df.to_csv(output_filename, sep="\t", header=False, index=False)
    print("Output File Written!")
   
d = distance.Distance()


def phonological_similarity(u_word, other_word):
    # print(f"Word1: {u_word} | Word2: {other_word}")

    return 0
    try:
        ed = editdistance.eval(u_word, other_word)
        eds = 0.01*1/ed
    except: 
        eds = 0

    try:
        fld = 0.01 *1/d.fast_levenshtein_distance_div_maxlen(u_word, other_word)
    except:
        fld = 0

    try:
        fer = 0.02 * 1/d.feature_error_rate([u_word], [other_word])
    except:
        fer = 0
        
    # return 0
    return fld + fer + eds



def semantic_similarity(u_word, u_gloss, u_emb, other_word, other_gloss, other_emb):

    # spacy_score = nlp(u_gloss).similarity(nlp(other_gloss))
    return np.dot(u_emb, other_emb) + int(u_gloss == other_gloss)



def scoring_function(u_word, u_gloss, u_emb, other_word, other_gloss, other_emb):
   score = phonological_similarity(u_word, other_word) + \
            semantic_similarity(u_word, u_gloss, u_emb, other_word, other_gloss, other_emb)

   return score


def get_top_5_words(u_word, u_gloss, u_emb, CAND_EP):
    # candidates = list(CAND_GLOSS_DIC.keys())
    

    scores = []
    # for c, cg in CAND_GLOSS_PAIRS:
    #     scores.append( (c, cg, scoring_function(u_word, u_gloss, c, cg)) )


    for (c, cg), ce in CAND_EP:
        scores.append( (c, cg, scoring_function(u_word, u_gloss, u_emb, c, cg, ce)) )

    scores = sorted(scores, key=lambda x:x[2], reverse=True)

    top_5 = scores[:5]
    print(f"    Word: {u_word} Gloss: {u_gloss} | Preds: {top_5}")


#    top_5 = candidates[:5]

    return [(t, tg) for t, tg, s in top_5]
   

# preds = [get_top_5_words(u) for u in inputs]

def evaluate(inputs, preds, GOLD_DIC):
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
    gold_cognate = GOLD_DIC['\t'.join(inp)]
    retrieved_cognates = ['\t'.join(p) for p in preds[idx]]
    if gold_cognate in retrieved_cognates:
      total.append(1.0 / (retrieved_cognates.index(gold_cognate) + 1))
  
  if len(total) == 0: return 0
  return sum(total) / len(total) 





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=int, help='0, 1 , or 2')
    args = parser.parse_args()

    languages = ["huishu", "kachai", "tusom"]
    language = languages[args.language]


    DATA_PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/dataset/"
    OUTPUT_PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/outputs/"

    CANDIDATE_PATH = f"{DATA_PATH}candidates/{language}_candidates.tsv"
    INPUT_PATH = f"{DATA_PATH}inputs/ukhrul-{language}_inputs.tsv"
    GOLD_PATH = f"{DATA_PATH}/ukhrul-huishu_gold.tsv"

    INP_GLOSS_PAIRS = read_df(INPUT_PATH)
    CAND_GLOSS_PAIRS = read_df(CANDIDATE_PATH)

    PATH = "/home/aaditd/4_Subword/P/subword-miniproject-4/embeddings/"


    emb1 = np.load(f'{PATH}ukhrul.npy')
    emb2 = np.load(f'{PATH}{language}.npy')

    INP_EP = list(zip(INP_GLOSS_PAIRS, emb1))
    CAND_EP = list(zip(CAND_GLOSS_PAIRS, emb2))



    print("DONE!")


    print(f"Unique Ukhrul Input words: {len(INP_GLOSS_PAIRS)}")
    print(f"Unique {language} Candidate words: {len(CAND_GLOSS_PAIRS)}")

    GOLD_DIC = read_gold_df(GOLD_PATH)
    # print()
    # # assert len(GOLD_DIC) == len(INP_GLOSS_PAIRS), f"Gold {len(GOLD_DIC)} and input file {len(INP_GLOSS_PAIRS)} have different number of entries!"


    test_inputs = INP_EP



    test_preds = [get_top_5_words(iw, ig, ie, CAND_EP) for (iw, ig), ie in test_inputs]
    # # for t in test_preds:
    # #     print(t)

    if language not in ["tusom", "kachai"]:
        print(evaluate(INP_GLOSS_PAIRS, test_preds, GOLD_DIC))
    write_output_df(test_preds, OUTPUT_PATH, language)


if __name__ == "__main__":
    main()


