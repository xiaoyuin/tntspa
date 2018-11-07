import sys


inp = sys.argv[1]
with open(inp) as f:
    results = list()
    for line in filter(lambda x: x.startswith('H'), f.readlines()):
        line = line.strip()
        head, score, sentence = line.split('\t')
        results.append((int(head.split('-')[1]), sentence))
    results.sort(key=lambda x: x[0])
    print(*[r[1] for r in results], sep='\n')
    
