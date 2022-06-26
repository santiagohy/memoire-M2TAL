import subprocess
import json
from pathlib import Path
from collections import namedtuple, defaultdict
from itertools import chain, combinations
import numpy as np
from typing import Dict
import csv
import re
from scipy.stats import fisher_exact

# ------------------ Fonctions ------------------

def conll_to_dict(path : str) -> dict:

    p = "../treebanks/" + path
    with open (p) as f:
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

def concat_patterns(patterns : tuple) -> str:
    res = "; ".join(patterns)
    res = f"pattern {{ {res} }}"
    return res

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def grew_count(*patterns , cluster : str, pattern_key : str, corpora) -> Dict[str, int]:
    subprocess.check_output(f"grew compile -i ../treebanks/{corpora}.json".split(), stderr=subprocess.STDOUT)
    with open("tmp.pat","w", encoding="utf-8") as file: 
        file.write(concat_patterns(patterns))
    command = ['grew', 'count', '-pattern', 'tmp.pat', f'-{cluster}', pattern_key, '-i', f'../treebanks/{corpora}.json']
    res_process = subprocess.run(command, capture_output=True, encoding='utf-8').stdout

    tmp_file = Path("tmp.pat").resolve()
    Path.unlink(tmp_file)
    
    freq_pat = {}
    for row in csv.DictReader(res_process.split("\n"), delimiter='\t'):
        freq_pat = {k : int(v) for k, v in row.items() if k != 'Corpus'}
    return freq_pat

# --------------------------------------------

# Variables
treebank = 'fr_gsd-sud.conllu'
corpora = "corpora-gsd"
filename = 'position_adj'
pattern_M1 = 'X[upos=NOUN]; Y[upos=ADJ]; X->Y'
pattern_M2 = 'Y << X'

# Parser le conllu
corpus = conll_to_dict(treebank)


# Récupérer toutes les phrases contenant le motif en utilisant Grew
pattern = 'pattern {' + pattern_M1 + '}'
with open("tmp.pat","w", encoding="utf-8") as f: 
    f.write(pattern)
    
command = f"grew grep -pattern tmp.pat -i ../treebanks/{treebank}".split()
match = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8").strip()
tmp_file = Path("tmp.pat").resolve()
Path.unlink(tmp_file)


# Récupérer tous les nœuds en question
lst_match = json.loads(match)
tpl_match = namedtuple('tpl_match', 'sent_id X Y')

to_recover = []
for m in lst_match:
    X = m['matching']['nodes']['X']
    Y = m['matching']['nodes']['Y']
    to_recover.append(tpl_match(m['sent_id'], X, Y))


# Création du dictionnaire avec les motifs et le nombre de accords et non-accords
patterns = defaultdict(lambda: defaultdict(int))


for tpl in to_recover:
    lst = []
    for i, token_id in enumerate([tpl.X, tpl.Y]):
        for k, v in corpus[tpl.sent_id][token_id].items():
            if k not in ("form", "upos", "lemma"):
                if k == "deprel" and tpl._fields[1+i] == "X":
                    continue
                elif k == "deprel":
                    lst.append(f"X-[{v}]->Y")
                elif k == "lemma" and tpl._fields[1+i] == "X":
                    continue
                else:
                    lst.append(f"{tpl._fields[1+i]}[{k}={v}]")

    if int(tpl.Y) < int(tpl.X):
        position = "yes"
    else:
        position = "no"
        
    combs = powerset(lst)
    for c in combs:
        if c:
            patterns[c][position] += 1


freq_cond = grew_count(pattern_M1, cluster="whether", pattern_key=pattern_M2, corpora=corpora)
# Le nombre total d'occurrences
M = sum([v for v in freq_cond.values()])

# Le nombre total d'accords
n = freq_cond['Yes']

# On calcule le test exact de Fisher pour chaque motif
res = []
for pat, value in patterns.items():
    N = sum([v for v in value.values()])
    k = value['yes']
    table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
    oddsr, pvalue = fisher_exact(table=table, alternative='greater')
    if pvalue < 0.01:
        percent_M1M2 = (k/n)*100
        percent_M1M3 = (k/N)*100
        probability_ratio = (k/N)/((n-k)/(M-N))    

        res.append([list(pat), pvalue, oddsr, probability_ratio, percent_M1M2, percent_M1M3])

res_sorted = sorted(res, key = lambda x: (x[1], -x[3]))
res_sorted_len = sorted(res, key = lambda x: len(x[0]))

# On filtre les résultats pour garder les motifs et sous-motifs plus significatifs
patterns, visited = set(), set()

for res in res_sorted_len:
    to_keep = set()
    pat, pvalue, _, PR, _, _ = res
    for xres in res_sorted_len:
        if set(pat).issubset(set(xres[0])):
            xpat, xpvalue, _, xPR, _, _ = xres
            if xpvalue < pvalue:
                to_keep.add(tuple(xpat))
            elif xpvalue == pvalue and xPR > PR:
                to_keep.add(tuple(xpat))
            elif xpvalue == pvalue and xPR == PR:
                if pat != xpat:
                    visited.add(tuple(xpat))
                else:
                    to_keep.add(tuple(pat))
            else:
                visited.add(tuple(xpat))
    patterns.update(to_keep)

patterns.difference_update(visited)

with open(f"../results/{filename}-{treebank.split('.')[0]}.txt", "w", encoding="utf-8") as f:
    f.write(f"pattern\tp-value\tPR\tpercentage k/M1&M2\tpercentage k/M1&M3\n")
    for lst in res_sorted:
        if tuple(lst[0]) in patterns:
            f.write(f"{lst[0]}\t{lst[1]}\t{lst[3]}\t{lst[4]}\t{lst[5]}\n")
