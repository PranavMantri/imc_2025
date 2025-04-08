import csv
import math
import networkx as nx


def detect_arbitrage(graph):
    nodes = list(graph.nodes)
    for source in nodes:
        # Bellman-Ford
        distances = {v: float('inf') for v in graph.nodes}
        predecessors = {v: None for v in graph.nodes}
        distances[source] = 0

        for _ in range(len(graph.nodes) - 1):
            for u, v, data in graph.edges(data=True):
                if distances[u] + data['weight'] < distances[v]:
                    distances[v] = distances[u] + data['weight']
                    predecessors[v] = u

        # Check for negative-weight cycle
        for u, v, data in graph.edges(data=True):
            if distances[u] + data['weight'] < distances[v]:
                print(f"Arbitrage opportunity detected starting from {source}!")
                cycle = [v]
                while True:
                    v = predecessors[v]
                    if v in cycle:
                        cycle = cycle[cycle.index(v):] + [v]
                        break
                    cycle.insert(0, v)
                print("Cycle:", " → ".join(cycle))
                return True
    print("No arbitrage opportunities found.")
    return False


