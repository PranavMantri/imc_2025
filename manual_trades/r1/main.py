import csv_graph
from arbitrage_detection import detect_arbitrage

def main():
    print("init graph")

    #inits -log weight di-graph from csv (networkx)
    g = csv_graph.read_exchange_csv("trading_table.csv")

    csv_graph.print_graph(g)

    detect_arbitrage(g)

if __name__ == "__main__":
    main()
