import grew
import subprocess
import json
from pathlib import Path
from collections import namedtuple, Counter
import numpy as np
from scipy.stats import fisher_exact

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
                    #trees[sent_id][token_id] = ({"form" : form, "lemma" : lemma, "upos" : upos, "deprel" : deprel})
                    trees[sent_id][token_id] = ({"upos" : upos, "deprel" : deprel})
                    if feats != "_":
                        features = [f.split("=") for f in feats.split("|")]
                        dict_features = {lst[0]:lst[1] for lst in features}
                        trees[sent_id][token_id].update(dict_features)
    return trees


# Variables
treebank = 'fr_gsd-sud.conllu'
feature = "Gender"

# Parser le conllu
corpus = conll_to_dict(treebank)

# Récupérer toutes les phrases contenant le motif en utilisant Grew
pattern = f'pattern {{X->Y; X[{feature}]; Y[{feature}]}}'
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

# Liste de dictionnaires avec les variables de chaque match
lst_predictors = []
for tpl in to_recover:
    dict_res = {}
    for k, v in corpus[tpl.sent_id][tpl.X].items():
        dict_res['X.'+k] = v
    for k, v in corpus[tpl.sent_id][tpl.Y].items():
        dict_res['Y.'+k] = v
    lst_predictors.append(dict_res)

# Combinaisons des succès et de non-succès
comb_agree = Counter([f"X[upos={lst['X.upos']}]; Y[upos={lst['Y.upos']}]; X-[{lst['Y.deprel']}]->Y" for lst in lst_predictors if lst[f'X.{feature}'] == lst[f'Y.{feature}']])
comb_desagree = Counter([f"X[upos={lst['X.upos']}]; Y[upos={lst['Y.upos']}]; X-[{lst['Y.deprel']}]->Y" for lst in lst_predictors if lst[f'X.{feature}'] != lst[f'Y.{feature}']])

# Nombre total de succès
n_agree = sum(comb_agree.values())
n = n_agree

# Nombre total de non-succès
n_desagree = sum(comb_desagree.values())

# Le nombre total d'occurrences
M = n_agree + n_desagree

# Compter les succès séparément par motif
dict = {}
for pattern, agree in comb_agree.items():
    try :
        desagree = comb_desagree[pattern]
    except KeyError:
        desagree = 0
    dict[pattern] = {"agree" : agree, "desagree" : desagree}

# On calcule le test exact de Fisher pour chaque motif
results = {}

for pat in dict:
    k = dict[pat]["agree"]
    N = k + dict[pat]["desagree"]
    table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
    oddsr, pvalue = fisher_exact(table=table, alternative='greater')
    percent_M1M2 = (k/n)*100
    percent_M1M3 = (k/N)*100
    probability_ratio = (k/N)/((n-k)/(M-N))
    results[pat] = {"pvalue" : pvalue, "oddsr" : oddsr, "probability_ratio" : probability_ratio, "percent_M1M2" : percent_M1M2, "percent_M1M3" : percent_M1M3}

sorted_results = sorted(results.items(), key=lambda x:x[1]["pvalue"],reverse=False)

with open(f"../results/accord-{feature}-triplets-{treebank.split('.')[0]}.txt", "w", encoding="utf-8") as f:
    f.write(f"pattern\tp-value\tprobability ratio\tpercentage k/M1&M2\t percentage k/M1&M3\n")
    for res in sorted_results:
        if res[1]['pvalue'] < 0.01:
            f.write(f"{res[0]}\t{res[1]['pvalue']}\t{res[1]['probability_ratio']}\t{res[1]['percent_M1M2']}\t{res[1]['percent_M1M3']}\n")
        else:
            f.write(f"{res[0]}\t{res[1]['pvalue']}\t{res[1]['probability_ratio']}\t{res[1]['percent_M1M2']}\t{res[1]['percent_M1M3']}\n")

