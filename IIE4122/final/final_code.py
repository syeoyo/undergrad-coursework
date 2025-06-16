import random
import time
from typing import Any, List, Tuple, Dict

class MultiplayerGame:
    def initial_state(self): pass
    def player(self, state): pass
    def actions(self, state): pass
    def result(self, state, action): pass
    def terminal(self, state): pass
    def utility(self, state): pass

def maxn(game: MultiplayerGame) -> Tuple[List[float], List[Any], List[Any]]:
    memo: Dict[Any, Tuple[List[float], List[Any], List[Any]]] = {}

    def search(state: Any) -> Tuple[List[float], List[Any], List[Any]]:
        if game.terminal(state):
            return game.utility(state), [], [state]
        if str(state) in memo:
            return memo[str(state)]
        p = game.player(state)
        best_util = None
        best_path = []
        best_states = []
        for a in game.actions(state):
            next_state = game.result(state, a)
            util, path, states = search(next_state)
            if (best_util is None) or (util[p] > best_util[p]):
                best_util = util
                best_path = [a] + path
                best_states = [state] + states
        memo[str(state)] = (best_util, best_path, best_states)
        return best_util, best_path, best_states

    return search(game.initial_state())

class RandomGame(MultiplayerGame):
    def __init__(self, num_players=3, depth=3, branch=2, seed=None):
        self.num_players = num_players
        self.depth = depth
        self.branch = branch
        self.tree = {}
        self.leaf_utilities = {}
        if seed is not None:
            random.seed(seed)
        self._generate_tree('root', 0)

    def _generate_tree(self, node, current_depth):
        if current_depth == self.depth:
            util = [random.randint(1, 10) for _ in range(self.num_players)]
            self.leaf_utilities[node] = util
            return
        self.tree[node] = []
        for i in range(self.branch):
            child = f"{node}_{i}"
            self.tree[node].append(child)
            self._generate_tree(child, current_depth + 1)

    def initial_state(self):
        return 'root'

    def player(self, state):
        if state == 'root':
            return 0
        return (state.count('_')) % self.num_players

    def actions(self, state):
        if state in self.tree:
            return list(range(len(self.tree[state])))
        return []

    def result(self, state, action):
        if state in self.tree:
            return self.tree[state][action]
        return state

    def terminal(self, state):
        return state in self.leaf_utilities

    def utility(self, state):
        return self.leaf_utilities.get(state, [0]*self.num_players)

if __name__ == '__main__':
    game = RandomGame(num_players=3, depth=3, branch=2, seed=42)
    start_time = time.time()
    util, path, states = maxn(game)
    exec_time = time.time() - start_time
    print('Optimal utility vector:', util)
    print('Optimal path (step by step):')
    for i in range(len(path)):
        player = states[i].count('_') % game.num_players if states[i] != 'root' else 0
        print(f"Step {i}: At node '{states[i]}', Player {player} chooses action {path[i]} -> '{states[i+1]}'")
    print(f"Final state: '{states[-1]}' with utility {util}")

    chosen_player = game.player(states[0])
    cost = util[chosen_player]
    print(f"Total cost (utility for first player): {cost}")
    print(f"Number of moves: {len(path)}")
    print(f"Execution time: {exec_time:.6f} seconds") 