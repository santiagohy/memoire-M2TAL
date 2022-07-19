import argparse, re
from collections import defaultdict, Counter
from itertools import combinations, chain
from typing import Dict
from scipy.stats import fisher_exact
import numpy as np
import grew

import time
# -------------------------

my_parser = argparse.ArgumentParser(description='Extraction of grammar rules from a treebank')

my_parser.add_argument('Treebank',
                       metavar='Path',
                       type=str,
                       help='the path to the conllu file')

my_parser.add_argument('-a',
                       '--all',
                       action='store_true',
                       help='an optional argument')

my_parser.add_argument('P1',
                       metavar='Pattern_P1',
                       type=str,
                       help='pattern P1')

my_parser.add_argument('P2',
                       metavar='Pattern_P2',
                       type=str,
                       help='pattern P2')

my_parser.add_argument('P3',
                       metavar='Pattern_P3',
                       type=str,
                       help='pattern P3')

args = my_parser.parse_args()

# ---------- Fonctions --------------------

def conllu2dict(path : str) -> Dict:

    with open (path, encoding="utf-8") as f:
        conll = f.read().strip()
        
    trees = dict()
    sentences = [x.split("\n") for x in conll.split("\n\n")]

    for sent in sentences:
        for line in sent:
            if line.startswith("#"):
                if "sent_id" in line:
                    sent_id = line.split("=")[1].strip()
                    trees[sent_id] = {}
                    trees[sent_id]['0'] = ({"form" : "None"})
            else:
                token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = line.split("\t")
                if "-" not in token_id:
                    trees[sent_id][token_id] = ({"form" : form, "lemma" : lemma, "upos" : upos, "deprel" : deprel})
                    if feats != "_":
                        features = [f.split("=") for f in feats.split("|")]
                        dict_features = {lst[0]:lst[1] for lst in features}
                        trees[sent_id][token_id].update(dict_features)
    return trees

def format_pattern(*pattern : str) -> str:
    res = ";".join(pattern)
    res = f"pattern {{ {res} }}"
    return res

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# ----------------------------------------

# Args
treebank_path = args.Treebank
option = args.all
P1 = args.P1
P2 = args.P2
P3 = args.P3


# Get predictors in a dictionary
predictors = defaultdict(list)

# Boolean to handle querys with or without keys
# For the moment the script doesn't accept mixed querys (with and without keys)

is_key = False

for s in P3.split(';'):
    if re.search(r"^.+?\.\w+?$", s):
        is_key = True
        k, v = s.strip().split(".")
        if v == "label":
            re_match = re.search(fr"{k}:(\w+?)->(\w+?)", P1)
            predictors[re_match.group(2)].append(["deprel", {"head" : re_match.group(1), "dep" : re_match.group(2)}])
        else:
            predictors[k].append(v)

# Load corpus using Grew
grew.init()
treebank_idx = grew.corpus(treebank_path)

# Load corpus in a dictionary
treebank = conllu2dict(treebank_path)
print("Corpus loaded!")

# Get nodes matching P1
matchs = grew.corpus_search(format_pattern(P1), treebank_idx)

# Getting combinations of predictors depeding on the is_key boolean

if is_key:
    res = []
    for m in matchs:
        lst = []
        for node, idx in m["matching"]["nodes"].items():
            for var in predictors[node]:
                # If it's a list is a deprel with a head and a dep
                if isinstance(var, list):
                    p = f"{var[1]['head']}-[{treebank[m['sent_id']][idx][var[0]]}]->{node}"
                else:
                    # Handling Node[Feature=Value]
                    try:
                        p = f"{node}[{var}={treebank[m['sent_id']][idx][var]}]"
                    except KeyError:
                        continue
                lst.append(p)
        # Handling option
        if option:
            # powerset
            values = ["; ".join(v) for v in powerset(lst) if v]
            for x in values:
                res.append(x)
        else:
            # largest combinations
            res.append("; ".join(lst))
    patterns = Counter(res)
# if is_key is False 
else:
    if option:
        patterns = ["; ".join(p) for p in powerset([x.strip() for x in P3.split(";")]) if p]
    else:
        patterns = [P3]

print("Combinations? Done!")
print("Significance calculation...")

# Significance calculation
M = grew.corpus_count(pattern = format_pattern(P1), corpus_index = treebank_idx)
n = grew.corpus_count(pattern = format_pattern(P1, P2), corpus_index = treebank_idx)

for pat in patterns:
    # this is for handle the two possible : if it's a dict it has key patterns, on the contrary it has only simple patterns
    if isinstance(patterns, dict):
        N = patterns[pat]
    else:
        N = grew.corpus_count(pattern = format_pattern(P1, pat), corpus_index = treebank_idx)
    k = grew.corpus_count(pattern = format_pattern(P1, P2, pat), corpus_index = treebank_idx)
    table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
    oddsr, p_value = fisher_exact(table = table, alternative='greater')
    if p_value < 0.01:
        print(pat, p_value)

toc = time.perf_counter()