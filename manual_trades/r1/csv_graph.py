import csv
import math
import networkx as nx

def read_exchange_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        headers = reader[0][1:]  # Skip the first empty cell
        data = reader[1:]        # Skip header row

        G = nx.DiGraph()

        for i, row in enumerate(data):
            from_currency = row[0]
            for j, cell in enumerate(row[1:]):
                to_currency = headers[j]
                try:
                    rate = float(cell)
                    if rate > 0:
                        weight = -math.log(rate)
                        G.add_edge(from_currency, to_currency, weight=weight, rate=rate)
                except ValueError:
                    pass  # Ignore non-numeric entries

        return G

def print_graph(graph):
    print("\n📊 Exchange Graph:")
    print("Nodes:")
    for node in graph.nodes:
        print(f"  - {node}")
    print("\nEdges:")
    for u, v, data in graph.edges(data=True):
        rate = data.get('rate', 'N/A')
        weight = data.get('weight', 'N/A')
        print(f"  {u} → {v} | rate: {rate}, weight: {weight:.4f}")

