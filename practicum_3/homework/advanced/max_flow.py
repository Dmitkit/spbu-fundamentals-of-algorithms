import networkx as nx
import copy

from typing import Any
from collections import deque

from src.plotting import plot_graph

# Проверям есть ли путь из s в t и сохраняем его
def bfs_residual(G: nx.Graph, s: Any, t: Any, parent):
    visited = {node: False for node in G.nodes}
    queue = deque()
    queue.append(s)
    visited[s] = True

    while queue:
        u = queue.popleft()
        for v in G.neighbors(u):
            if not visited[v] and G[u][v]['weight'] > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == t:
                    return True
    return False

def max_flow(G: nx.Graph, s: Any, t: Any) -> int:
    parent = {node: -1 for node in G.nodes}
    max_flow = 0

    while bfs_residual(G, s, t, parent):
        path_flow = float("inf")
        n = t
        while n != s:
            path_flow = min(path_flow, G[parent[n]][n]['weight'])
            n = parent[n]
        max_flow += path_flow

        v = t
        while v != s:
            u = parent[v]
            G[u][v]['weight'] -= path_flow
            v = u

    return max_flow


if __name__ == "__main__":
    G = nx.read_edgelist("practicum_3/homework/advanced/graph_1.edgelist", create_using=nx.DiGraph, nodetype=int)
    G_copy = copy.deepcopy(G)

    val = max_flow(G_copy, s=0, t=5)
    print(f"Maximum flow is {val}. Should be 23")
    plot_graph(G)
