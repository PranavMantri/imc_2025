import itertools

# Define the exchange rates as a dictionary
exchange_rates = {
    'Snowballs': {'Snowballs': 1, 'Pizzas': 1.45, 'Silicon Nuggets': 0.52, 'SeaShells': 0.72},
    'Pizzas': {'Snowballs': 0.7, 'Pizzas': 1, 'Silicon Nuggets': 0.31, 'SeaShells': 0.48},
    'Silicon Nuggets': {'Snowballs': 1.95, 'Pizzas': 3.1, 'Silicon Nuggets': 1, 'SeaShells': 1.49},
    'SeaShells': {'Snowballs': 1.34, 'Pizzas': 1.98, 'Silicon Nuggets': 0.64, 'SeaShells': 1}
}

currencies = list(exchange_rates.keys())  # Use the keys from the dictionary to avoid typos

def find_all_trades():
    all_trades = []

    for path_length in range(2, 6):  # lengths from 2 to 5
        for intermediate in itertools.product(currencies, repeat=path_length - 2):
            full_path = ('SeaShells',) + intermediate + ('SeaShells',)

            product = 1.0
            valid_path = True
            for i in range(len(full_path) - 1):
                from_curr = full_path[i]
                to_curr = full_path[i + 1]
                if to_curr not in exchange_rates[from_curr]:
                    valid_path = False
                    break
                product *= exchange_rates[from_curr][to_curr]

            if valid_path:
                profit = product - 1
                all_trades.append((profit, full_path))

    # Sort all trades by profit
    all_trades.sort(reverse=True, key=lambda x: x[0])
    return all_trades

# Write to file
trades = find_all_trades()
output_file = "all_trades.txt"
with open(output_file, "w") as f:
    f.write("All trades from SeaShells to SeaShells (sorted by profit):\n\n")
    for i, (profit, path) in enumerate(trades, 1):
        f.write(f"{i}. Path: {' -> '.join(path)}\n")
        f.write(f"   Profit: {profit:.6f} (1 SeaShell -> {1 + profit:.6f} SeaShells)\n\n")

print(f"All trades written to {output_file}")
