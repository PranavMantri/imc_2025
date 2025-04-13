import numpy as np
from collections import defaultdict


class Container:
    def __init__(self, index, base_price, multiplier, inhabitants=0, probability=0.1):
        self.index = index
        self.base_price = base_price
        self.multiplier = multiplier
        self.inhabitants = inhabitants
        self.players = 0
        self.probability = probability
        self.avg_payoff = 0

    def expected_payoff(self, total_players):
        denom = self.inhabitants + 100*(self.players / total_players if total_players > 0 else 0)
        return (self.base_price * self.multiplier) / max(denom, 1e-5)


class Agent:
    def __init__(self, strategy=None):
        self.strategy = strategy
        self.payoff = 0

    def assign_payoff(self, containers_dict, total_players):
        self.payoff = 0
        for idx in self.strategy:
            c = containers_dict[idx]
            denom = c.inhabitants + 100*(c.players / total_players if total_players > 0 else 0)
            self.payoff += (c.base_price * c.multiplier) / max(denom, 1e-5)
        if len(self.strategy) == 2:
            self.payoff -= 50000


def print_game_state(containers, total_players):
    best_idx = None
    best_value = -float('inf')
    expected_values = {}

    for idx, c in containers.items():
        payoff = c.expected_payoff(total_players)
        expected_values[idx] = payoff

        if payoff > best_value:
            best_value = payoff
            best_idx = idx

        f = open("game_state.txt", 'a')
        f.write(f"{idx},{c.multiplier},{c.inhabitants},{c.players},{c.probability},{payoff}\n")
        f.close()


def update_probabilities(containers, beta=0.2):
    """
    Update container probabilities in-place.
    beta ∈ [0,1] is a learning rate controlling responsiveness to payoff.
    """
    container_list = list(containers.values())
    n = len(container_list)
    if n == 0:
        return

    payoffs = np.array([c.avg_payoff for c in container_list])
    avg_payoff = np.mean(payoffs)

    # Handle degenerate case
    if avg_payoff <= 0:
        for c in container_list:
            c.probability = 1.0 / n
        return

    # Offset if needed
    min_payoff = np.min(payoffs)
    offset = -min_payoff + 1e-9 if min_payoff < 0 else 0
    adjusted = payoffs + offset
    adjusted_avg = avg_payoff + offset

    # Compute new weights (just based on payoff)
    raw_weights = adjusted / adjusted_avg
    weight_sum = np.sum(raw_weights)

    for c, w in zip(container_list, raw_weights):
        target_prob = w / weight_sum
        # Blend with previous probability to smooth changes
        c.probability = (1 - beta) * c.probability + beta * target_prob

    # Final normalize to ensure they sum to 1
    total_prob = sum(c.probability for c in container_list)
    for c in container_list:
        c.probability /= total_prob




def run_simulation(rounds, num_agents=9000, beta=0.1, explore_rate=0.4):
    containers = {
        i: Container(i, base_price=10000, multiplier=m, inhabitants=initial)
        for i, (m, initial) in enumerate([
            (10, 1), (80, 6), (37, 3), (17, 1), (90, 10),
            (31, 2), (50, 4), (20, 2), (73, 4), (89, 8)
        ])
    }

    # Persistent agents across rounds
    agents = [Agent() for _ in range(num_agents)]

    for r in range(rounds):
        for c in containers.values():
            c.players = 0
            c.avg_payoff = 0

        probs = np.array([c.probability for c in containers.values()])
        keys = np.array(list(containers.keys()))
        nonzero_mask = probs > 1e-8
        filtered_probs = probs[nonzero_mask]
        filtered_probs /= filtered_probs.sum()
        filtered_keys = keys[nonzero_mask]

        for agent in agents:
            if agent.strategy is None or np.random.rand() < 1:
                pick_count = np.random.choice([1, 2], p=[0.5, 0.5])
                if len(filtered_keys) < pick_count:
                    # Fallback: sample fewer or allow replacement
                    print("wut")
                    return
                else:
                    strategy = tuple(np.random.choice(filtered_keys, size=pick_count, replace=False, p=filtered_probs))

                agent.strategy = strategy

            for idx in agent.strategy:
                containers[idx].players += 1

        total_players = sum(c.players for c in containers.values())

        choice_payoffs = defaultdict(list)
        for agent in agents:
            agent.assign_payoff(containers, total_players)
            choice_payoffs[agent.strategy].append(agent.payoff)

        avg_choice_payoff = {
            choice: np.mean(payoffs) for choice, payoffs in choice_payoffs.items()
        }
        sorted_choices = sorted(avg_choice_payoff.items(), key=lambda x: -x[1])



        for c in containers.values():
            relevant_agents = [a for a in agents if c.index in a.strategy]
            if relevant_agents:
                c.avg_payoff = np.mean([a.payoff for a in relevant_agents])
            else:
                c.avg_payoff = 0

        if(r  > 0):
            print_game_state(containers, num_agents)

            f = open("game_choices.txt", 'a')
            f.write(f"\nRound {r + 1}")
            for i, (choice, payoff) in enumerate(sorted_choices[:5]):
                pretty_choice = tuple(int(x) for x in choice)
                f.write(f"{i + 1:>2},{pretty_choice},{payoff:,.2f}")
            f.close()
        update_probabilities(containers, beta=beta)


    return containers


def main():

    containers = run_simulation(rounds=1000)


if __name__ == "__main__":
    main()
