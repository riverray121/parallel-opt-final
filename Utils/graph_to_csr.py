"""
Convert edge-list graphs (i.e., datasets from the internet) into CSR used by the search algorithm
"""

import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python convert_to_csr.py <graph_file>")
    sys.exit(1)

graph_file = sys.argv[1]

print(f"Loading graph from {graph_file}...")
G = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.DiGraph)
csr = csr_matrix(nx.to_scipy_sparse_array(G, format='csr'))

row_offsets = csr.indptr
np.savetxt("row_offsets.txt", row_offsets, fmt="%d")

column_indices = csr.indices
np.savetxt("column_indices.txt", column_indices, fmt="%d")

print("Conversion complete.")
print("Row offsets saved to: row_offsets.txt")
print("Column indices saved to: column_indices.txt")