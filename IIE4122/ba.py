from math import exp, sin
import numpy as np

class BeeAlgorithm:
    def __init__(
        self,
        distances,
        directions, 
        communications,
        n_bees=20,
        n_best=10,
        num_iterations=100,
        alpha=1,
        beta=2,
        gamma=0.1,
    ):
        self.distances = distances
        self.directions = directions
        self.communications = communications
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_bees = n_bees
        self.n_best = n_best
        self.n_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for _ in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, self.gamma)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
                visited = set()
                visited.add(shortest_path[0][0][0])
                for frm, to in shortest_path[0]:
                    visited.add(to)
                skipped = [i for i in self.all_inds if i not in visited]
                print(f"Visited cities: {sorted(list(visited))}")
                print(f"Skipped cities: {sorted(skipped)}")
                print(f"Shortest Path: {shortest_path[0]}")
                print(f"Total Cost: {shortest_path[1]}")
        return all_time_shortest_path
    
    def spread_pheronome(self, all_paths, n_best, gamma):
        self.pheromone *= (1 - self.gamma)
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, _ in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += exp(-self.distances[move]) / (
                    1 + sin(self.distances[move]) ** 2
                ) 

    def gen_path_dist(self, path):
        total_distance = 0
        for ele in path:
            total_distance += self.distances[ele]
        return total_distance

    def gen_all_paths(self):
        all_paths = []
        for _ in range(self.n_bees):
            path = self.waggle_dance(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths
    
    def waggle_dance(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for _ in range(len(self.distances) - 1):
            communicate = self.communications
            move = self.pick_move(self.pheromone[prev], self.distances[prev], self.directions[prev], visited, communicate)
            if move is None:
                break
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        skipped = [i for i in self.all_inds if i not in visited]
        return path

    def pick_move(self, pheromone, dist, direction_row, visited, communicate):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        candidates = []
        for i in self.all_inds:
            if (
                i not in visited
                and communicate[i] == 0 
                and direction_row[i] == 1
            ):
                candidates.append(i)

        if not candidates:
            return None

        min_cost = np.inf
        selected = None
        for i in candidates:
            cost = dist[i]
            if cost < min_cost:
                min_cost = cost
                selected = i
        return selected
    
if __name__ == "__main__":
    distances = np.array(
        # Seoul(0) / Gangneung(1) / Busan(2) / Daejeon(3) / Mokpo(4)
        [
            [np.inf, 165, 325, 140, 300],
            [165, np.inf, 290, 200, 200],
            [325, 290, np.inf, 200, 250],
            [140, 200, 200, np.inf, 190],
            [300, 400, 250, 190, np.inf],
        ]
    )
    directions_included = np.array(
        # "1" means path available, "0" means path unavailable
       [
           [0,  1,  1,  0,  1],
           [1,  0,  1,  1,  1],
           [1,  1,  0,  1,  1],
           [1,  1,  1,  0,  0],
           [1,  0,  1,  1,  0],
       ] 
    )
    directions_excluded = np.array(
       [
           [0,  1,  1,  1,  1],
           [1,  0,  1,  1,  1],
           [1,  1,  0,  1,  1],
           [1,  1,  1,  0,  1],
           [1,  1,  1,  1,  0],
       ] 
    )
    
    communications_included = np.array([0, 1, 0, 0, 0])
    communications_excluded = np.array([0, 0, 0, 0, 0])
    # 0 means "OK to visit", 1 means "Do NOT visit this city"
    
    print("DIRECTION EXCLUDED/COMMUNICATION EXCLUDED = ANT COLONY OPTIMIZATION")
    bee_algorithm_1 = BeeAlgorithm(distances, directions_excluded, communications_excluded, n_bees=20, n_best=10, num_iterations=100, alpha=1, beta=2)
    bee_algorithm_1.run()
    print("")
    
    print("DIRECTION INCLUDED/COMMUNICATION EXCLUDED")
    bee_algorithm_2 = BeeAlgorithm(distances, directions_included, communications_excluded, n_bees=20, n_best=10, num_iterations=100, alpha=1, beta=2)
    bee_algorithm_2.run()
    print("")
    
    print("DIRECTION INCLUDED/COMMUNICATION INCLUDED = BEE ALGORITHM")
    bee_algorithm_3 = BeeAlgorithm(distances, directions_included, communications_included, n_bees=20, n_best=10, num_iterations=100, alpha=1, beta=2)
    bee_algorithm_3.run()