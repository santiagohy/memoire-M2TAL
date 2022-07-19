import re
from collections import defaultdict, Counter
from itertools import combinations, chain
from typing import Dict, Tuple
from scipy.stats import fisher_exact
import numpy as np
import grew
from pathlib import Path
import pandas as pd

# ---------- Fonctions ----------

def load_corpus(content : str, filename : str) -> Tuple[int, Dict]:

    # Load corpus using Grew
    with open (filename, "wb") as f:
        f.write(content)

    treebank_idx = grew.corpus(filename)
    # Load corpus in a dictionary
    treebank = conllu2dict(filename)

    # Erase file
    f = Path(filename).resolve()
    Path.unlink(f)

    return treebank_idx, treebank


def conllu2dict(path : str) -> Dict:

    with open (path) as f:
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

def format_significance(p_value : float) -> int:
    if "-" in str(p_value):
        significance = str(p_value).split("-")[1]
        significance = int(re.sub(r"^0", "", significance))
    elif p_value == 0:
        significance = np.inf
    else:
        significance = int(re.search(r"\.(0+)", str(p_value)).group(1).count("0"))
    return significance 

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_predictors(P1, P3):
    # Get predictors in a dictionary
    any_key = False
    predictors = defaultdict(list)

    # Boolean to handle querys with or without keys
    # For the moment the script doesn't accept mixed querys (with and without keys)

    for s in P3.split(';'):
        if re.search(r'^.+?\.\w+?$', s):
            any_key = True
            k, v = s.strip().split(".")
            if v == "label":
                re_match = re.search(fr"{k}:(\w+?)->(\w+?)", P1)
                predictors[re_match.group(2)].append(["deprel", {"head" : re_match.group(1), "dep" : re_match.group(2)}])
            else:
                predictors[k].append(v)
    return predictors, any_key  

def get_patterns2query(treebank_idx, treebank, P1, P3, option):

    predictors, any_key = get_predictors(P1, P3)

    # Get nodes matching P1
    matchs = grew.corpus_search(format_pattern(P1), treebank_idx)

    # Getting combinations of predictors depeding on the any_key boolean
    if any_key:
        res = []
        for m in matchs:
            lst = []
            for node, idx in m["matching"]["nodes"].items():
                for var in predictors[node]:
                    # If it's a list is a deprel with a head and a dep
                    if isinstance(var, list):
                        p = f'{var[1]["head"]}-[{treebank[m["sent_id"]][idx][var[0]]}]->{node}'
                    else:
                        # Handling Node[Feature=Value]
                        try:
                            p = f'{node}[{var}="{treebank[m["sent_id"]][idx][var]}"]'
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
                if lst:
                    res.append("; ".join(lst))
        patterns = Counter(res)
    # if is_key is False/None 
    else:
        if option:
            patterns = ["; ".join(p) for p in powerset([x.strip() for x in P3.split(";")]) if p]
        else:
            patterns = [P3]
    return patterns


def rules_extraction(treebank_idx, treebank, P1, P2, P3, option):

    res = []

    patterns = get_patterns2query(treebank_idx, treebank, P1, P3, option)
    # Significance calculation
    M = grew.corpus_count(pattern = format_pattern(P1), corpus_index = treebank_idx)
    n = grew.corpus_count(pattern = format_pattern(P1, P2), corpus_index = treebank_idx)
    for pat in patterns:
        # this is for handle the two possibilities : if it's a dict it has key patterns, on the contrary it has only simple patterns
        if isinstance(patterns, dict):
            N = patterns[pat]
        else:
            N = grew.corpus_count(pattern = format_pattern(P1, pat), corpus_index = treebank_idx)

        k = grew.corpus_count(pattern = format_pattern(P1, P2, pat), corpus_index = treebank_idx)
        table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
        _, p_value = fisher_exact(table = table, alternative='greater')
        if p_value < 0.01:
            significance = format_significance(p_value)
            percent_M1M2 = round((k/n)*100, 2)
            percent_M1M3 = round((k/N)*100, 2)
            probability_ratio = round((k/N)/((n-k)/(M-N)), 2)
            res.append([pat, significance, probability_ratio, percent_M1M2, percent_M1M3])
    
    df = pd.DataFrame(res, columns=["Pattern", "Significance", "Probability ratio", "% of P1&P2", "% of P1&P3"])
    return df