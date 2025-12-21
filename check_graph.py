import pickle
from collections import Counter

with open('outputs/gpt-4o_text-embedding-3-large/graph.pickle', 'rb') as f:
    graph = pickle.load(f)

print('Node attributes:', graph.vs.attributes())
print('Edge attributes:', graph.es.attributes())
print(f'Total nodes: {graph.vcount()}')
print(f'Total edges: {graph.ecount()}')
print()

# Check hash_id prefixes to understand node types
prefixes = Counter()
for v in graph.vs:
    hash_id = v['hash_id']
    prefix = hash_id.split('-')[0] if '-' in hash_id else 'unknown'
    prefixes[prefix] += 1

print('Node types by hash_id prefix:')
for prefix, count in prefixes.items():
    print(f'  {prefix}: {count}')

# Sample edges to see connections
print()
print('Sample edges (first 10):')
for e in graph.es[:10]:
    src = graph.vs[e.source]
    tgt = graph.vs[e.target]
    print(f'  "{src["content"]}" --[{e["weight"]:.4f}]--> "{tgt["content"]}"')
