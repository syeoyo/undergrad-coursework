from math import exp, sin
import numpy as np

class BeeAlgorithm:
    def __init__(
        self,
        distances,
        directions, 
        communication,
        n_bees=20,
        n_best=10,
        num_iterations=100,
        alpha=1,
        beta=2,
        gamma=0.1,
    ):
        self.distances = distances
        self.directions = directions
        self.communication = communication
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
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        min_distance = np.inf
        move = None
        for i in self.all_inds:
            if i not in visited and dist[i] < min_distance:
                min_distance = dist[i]
                move = i
        return move

    def waggle_dance(self):
        # strength(scalar = ACO와 동일하게)
        # direction(vector, ACO와 차이점1) - vector matrix 추가해야함
        # communication with other salesman(broadcasting, ACO와 차이점2)
        # one extra loop will be added - (if communicate "yes selling", then visit city / else (=communicate "no selling in Daehu") then skip city
        # if city is skipped, show it is skipped
        # if skip, print result (path and cost) AND if not skip, print result (path and cost) -> show both
        # if no skip = identical as ACO
        # cost vector를 만들으라는 소리인가
        return 

"""
Seoul(0)   Gangneung(1)   Busan(2)   Daejeon(3)   Mokpo(4)
 np.inf        165          325         140         300
  165         np.inf        290         200         400
  325          290        np.inf        200         250
  140          200          200       np.inf        190
  300          400          250         190        np.inf
"""

if __name__ == "__main__":
    distances = np.array(
        [
            [np.inf, 165, 325, 140, 300],
            [165, np.inf, 290, 200, 200],
            [325, 290, np.inf, 200, 250],
            [140, 200, 200, np.inf, 190],
            [300, 400, 250, 190, np.inf],
        ]
    )
    directions = np.array(
       [
           [], 
           [],
           [],
           [],
           [],
       ] 
    )
    communications = np.array(
        [
            [], 
            [],
            [],
            [],
            [],
        ]
    )
    bee_algorithm = BeeAlgorithm(
        distances, directions, communications, n_bees=20, n_best=10, num_iterations=100, alpha=1, beta=2
    )
    shortest_path = bee_algorithm.run()
    print(f"Shortest Path: {shortest_path}")
