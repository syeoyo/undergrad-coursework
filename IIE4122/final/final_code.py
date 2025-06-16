import time, random

class Node:
    def __init__(self, name, player, children=None, utility=None):
        self.name, self.player, self.children, self.utility = name, player, children or [], utility
    is_terminal = lambda self: self.utility is not None

next_player = lambda p: (p + 1) % 3
random_utility = lambda: tuple(random.randint(1, 100) for _ in range(3))

def build_tree(depth, max_depth, player, idx):
    name = f"{depth}_{bin(idx)[2:]}"
    if depth == max_depth:
        return Node(name, None, utility=random_utility())
    next_p = next_player(player)
    return Node(name, player, [build_tree(depth+1, max_depth, next_p, idx*2), build_tree(depth+1, max_depth, next_p, idx*2+1)])

root = build_tree(0, 9, 0, 1)

class Game:
    def __init__(self, root):
        self.current, self.history, self.nodes_explored = root, [], 0

    def minimax(self, node, alpha, beta, maximizing_player, state):
        self.nodes_explored += 1
        if node.is_terminal():
            return node.utility
        player = node.player
        best_utility = None
        for child in node.children:
            next_state = {'moves': state['moves']+1, 'path': state['path']+[child.name]}
            utility = self.minimax(child, alpha, beta, player, next_state)
            if best_utility is None or utility[player] > best_utility[player]:
                best_utility = utility
            if player == maximizing_player:
                alpha = max(alpha, best_utility[player])
            else:
                beta = min(beta, best_utility[player])
            if beta <= alpha:
                break
        return best_utility

    def play(self):
        move_num = 1
        def evaluate_child(child, player):
            self.nodes_explored = 0
            state = {'moves': len(self.history), 'path': self.history + [child.name]}
            utility = self.minimax(child, float('-inf'), float('inf'), player, state)
            return child, utility, self.nodes_explored
        while not self.current.is_terminal():
            player = self.current.player
            player_label = chr(65 + player)
            print(f"[TURN {move_num}] Player {player_label}'s turn (Current Node: {self.current.name})")
            start_time = time.time()
            results = [evaluate_child(child, player) for child in self.current.children]
            best_child, best_utility, best_nodes_explored = max(
                results,
                key=lambda x: (x[1][player], -x[2])
            )
            elapsed_ms = (time.time() - start_time) * 1000
            print(f" --> Player {player_label} chooses: {best_child.name}")
            print(f" Step execution time: {elapsed_ms:.3f} ms")
            print(f" Nodes exploring cost: {best_nodes_explored}")
            print(f" Expected utility: A = {best_utility[0]}, B = {best_utility[1]}, C = {best_utility[2]}\n")
            self.history.append(self.current.name)
            self.current = best_child
            move_num += 1

        final_utility = self.current.utility
        winner_idx = max(range(3), key=lambda i: final_utility[i])
        winner_label = chr(65 + winner_idx)
        print("[GAME END]")
        print(f"Final Node: {self.current.name}")
        print(f"Final Utility: A = {final_utility[0]}, B = {final_utility[1]}, C = {final_utility[2]}")
        print(f"Winner: Player {winner_label} with utility {final_utility[winner_idx]}")
        print(f"Path: {' -> '.join(self.history + [self.current.name])}")

Game(root).play()
