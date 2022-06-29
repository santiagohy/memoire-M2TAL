from grew import grew, corpus_count
import numpy as np
import subprocess
from pathlib import Path
import re
from numpy import inf
from scipy.stats import fisher_exact

def concat_patterns(*pattern : str) -> str:
    res = ";".join(pattern)
    res = f"pattern {{ {res} }}"
    return res
    
grew.init()

# Variables
conll = "../treebanks/fr_gsd-sud.conllu"
corpora = "corpora-gsd"
name = "sujet_inv"

M1 = 'H->X; X-[subj]->Y; Y[upos=NOUN|PROPN]'
M2 = "X << Y"
M3 = "e:H->X"


# Indexation du corpus en utilisant grew
corpus_index = grew.corpus(conll)

# Récupérer toutes les phrases contenant le motif en utilisant Grew
with open("tmp.pat","w", encoding="utf-8") as file: 
    file.write(concat_patterns(M1, M2, M3))
subprocess.check_output(f"grew compile -i ../treebanks/{corpora}.json".split(), stderr=subprocess.STDOUT)
command = f"grew count -pattern tmp.pat -key e.label -i ../treebanks/{corpora}.json".split()
tsv = subprocess.check_output(command, stderr=subprocess.STDOUT).decode("utf-8").strip()
tmp_file = Path("tmp.pat").resolve()
Path.unlink(tmp_file)  

# Récupérer les valeurs fixes via Grew
M3_patterns = tsv.split("\n")[0].split("\t")[1:]
k_values = tsv.split("\n")[1].split("\t")[1:]
basic_M3 = re.sub(r"^e:", "", M3)

M_pattern = concat_patterns(M1)
n_pattern = concat_patterns(M1, M2)

M = corpus_count(pattern = M_pattern, corpus_index = corpus_index)
n = corpus_count(pattern = n_pattern, corpus_index = corpus_index)


# On calcule le test exact de Fisher pour chaque motif
results = dict()

for m, k in zip(M3_patterns, k_values):

    M3 = basic_M3
    M3 = re.sub(r"->", f"-[{m}]->", M3)
    N_pattern = concat_patterns(M1, M3)
    N = corpus_count(pattern = N_pattern, corpus_index = corpus_index)
    k = int(k)

    table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
    oddsr, p_value = fisher_exact(table = table, alternative='greater')
    percent_M1M2 = (k/n)*100
    percent_M1M3 = (k/N)*100
    probability_ratio = (k/N)/((n-k)/(M-N))

    results[m] = {"pvalue" : p_value, "oddsr" : oddsr, "probability_ratio" : probability_ratio, "percent_M1M2" : percent_M1M2, "percent_M1M3" : percent_M1M3}

sorted_results = sorted(results.items(), key=lambda x:x[1]["pvalue"],reverse=False)

with open(f"../results/{name}-{conll.split('.')[0]}.txt", "w", encoding="utf-8") as f:
    f.write(f"pattern\tp-value\tprobability ratio\tpercentage k/M1&M2\t percentage k/M1&M3\n")
    for res in sorted_results:
        if res[1]['pvalue'] < 0.01:
            f.write(f"+\t{res[0]}\t{res[1]['pvalue']}\t{res[1]['probability_ratio']}\t{res[1]['percent_M1M2']}\t{res[1]['percent_M1M3']}\n")
        else:
            f.write(f"-\t{res[0]}\t{res[1]['pvalue']}\t{res[1]['probability_ratio']}\t{res[1]['percent_M1M2']}\t{res[1]['percent_M1M3']}\n")
