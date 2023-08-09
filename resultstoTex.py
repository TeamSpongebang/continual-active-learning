import numpy as np
import sys

path, n_episode = sys.argv[1:]
n_episode = int(n_episode)

with open(path) as f:
    results = f.readlines()

matrix = [[-1 for _ in range(n_episode)] for _ in range(n_episode)]

for train_stream in range(n_episode):
    for test_stream in range(n_episode):
        acc = float(results[1 + train_stream * (n_episode + 2) + test_stream + 1].replace(',',''))
        matrix[train_stream][test_stream] = acc

with open(path+".tex", 'w') as f:
    tex = r"\begin{bmatrix}"
    
    for row in matrix:
        tex += "&".join([f'{100 * item:.2f}' for item in row]) + "\\\\"
    
    tex += r"\end{bmatrix}"
    
    f.write(tex)

matrix = np.array(matrix)
matrix = matrix[1:,1:]

backward = 0
in_domain = 0
next_domain = 0
forward = 0

for idx, row in enumerate(matrix):
    backward += row[:idx].sum()
    forward += row[idx+1:].sum()
    in_domain += row[idx]
    if idx < len(matrix) - 1:
        next_domain += row[idx+1]

backward /= 15
forward /= 15
next_domain /= 5
in_domain /= 6

performances = {"in_domain": in_domain, "next_domain": next_domain, "backward": backward, "forward": forward}
performances.update({"matrix": matrix.tolist()})
import json
with open(path+".json","w") as f:
    json.dump(performances, f, indent=4)