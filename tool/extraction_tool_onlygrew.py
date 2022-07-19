
from pathlib import Path
from scipy.stats import fisher_exact
import grew
from typing import Dict
import argparse, csv, re, subprocess
from itertools import product, combinations
import numpy as np

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

my_parser.add_argument('Corpora',
                       metavar='Corpora',
                       type=str,
                       help='corpora json')

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

# ------------------------------

def format_pattern(*pattern : str) -> str:
    res = ";".join(pattern)
    res = f"pattern {{ {res} }}"
    return res

# Un problème : comment compiler le json de façon automatique
def grew_count(*patterns , cluster : str, key_pattern : str, corpora) -> Dict[str, int]:

    #subprocess.check_output(f"grew compile -i ../treebanks/{corpora}.json".split(), stderr=subprocess.STDOUT)

    with open("tmp.pat","w", encoding="utf-8") as file: 
        file.write(format_pattern(*patterns))
    command = ['grew', 'count', '-pattern', 'tmp.pat', f'-{cluster}', key_pattern, '-i', corpora]
    res_process = subprocess.run(command, capture_output=True, encoding='utf-8').stdout

    # Erase file
    tmp_file = Path("tmp.pat").resolve()
    Path.unlink(tmp_file)
    
    for row in csv.DictReader(res_process.split("\n"), delimiter='\t'):
        key_res = [k for k in row.keys() if k != 'Corpus']
    return key_res


treebank_path = args.Treebank
option = args.all
corpora = args.Corpora
P1 = args.P1
P2 = args.P2
P3 = [s.strip() for s in args.P3.split(';')]

preds = []

# grew_count + key to get the values
for pat in P3:
    if ".label" in pat:
        label = pat.split(".")[0]
        m = re.search(fr"{label}:(\w+?)->(\w+?)", P1)
        deprels = grew_count(P1, cluster="key", key_pattern = pat, corpora = corpora)
        preds.append([f"{m.group(1)}-[{d}]->{m.group(2)}" for d in deprels])
    elif ("." in pat and "=" not in pat):
        node, feat = pat.split(".")
        features = grew_count(P1, cluster="key", key_pattern = pat, corpora = corpora)
        preds.append([f"{node}[{feat}={f}]" for f in features])
    else:
        preds.append([pat])

if option:
    # Get all possible combinations
    patterns = [p for r in range(len(preds)) for c in combinations(preds, r + 1) for p in product(*c)]
else:
    # Get all the largest combinations
    patterns = list(product(*preds))

grew.init()
treebank_idx = grew.corpus(treebank_path)

M = grew.corpus_count(pattern = format_pattern(P1), corpus_index = treebank_idx)
n = grew.corpus_count(pattern = format_pattern(P1, P2), corpus_index = treebank_idx)

for c in patterns:
    P3 = "; ".join(c)
    N = grew.corpus_count(pattern = format_pattern(P1, P3), corpus_index = treebank_idx)
    k = grew.corpus_count(pattern = format_pattern(P1, P2, P3), corpus_index = treebank_idx)
    table = np.array([[k, n-k], [N-k, M - (n + N) + k]])
    oddsr, p_value = fisher_exact(table = table, alternative='greater')
    if p_value < 0.01:
        print(P3, p_value)
