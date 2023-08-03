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
        
import pdb;pdb.set_trace()

with open(path+".tex", 'w') as f:
    tex = r"\begin{bmatrix}"
    
    for row in matrix:
        tex += "&".join([f'{100 * item:.2f}' for item in row]) + "\\\\"
    
    tex += r"\end{bmatrix}"
    
    f.write(tex)