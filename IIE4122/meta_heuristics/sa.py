# Traveling Salesman Problem using Simulated Annealing Algorithm

import random
import numpy as np

class SA:
    def __init__(self, num_city, data):
        self.T0 = 1500
        self.Tf = 1e-6
        self.cooling_rate = 0.995
        self.max_iter = 10000
        self.num_city = num_city
        self.location = data
        self.distance_matrix = self.compute_distance_matrix(num_city, data)
        self.current_path = self.random_init(num_city)

    def random_init(self, num_city):
        tmp = [x for x in range(num_city)]
        random.shuffle(tmp)
        return tmp

    def compute_distance_matrix(self, num_city, location):
        distance_matrix = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    distance_matrix[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                distance_matrix[i][j] = tmp
        return distance_matrix

    def compute_total_distance(self, path, distance_matrix):
        a = path[0]
        b = path[-1]
        result = distance_matrix[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += distance_matrix[a][b]
        return result

    def get_new_path(self, path):
        path = path.copy()
        t = [x for x in range(len(path))]
        a, b = np.random.choice(t, 2)
        path[a:b] = path[a:b][::-1]
        return path

    def eval_fire(self, current_best, candidate, temp):
        current_best_length = self.compute_total_distance(current_best, self.distance_matrix)
        new_length = self.compute_total_distance(candidate, self.distance_matrix)
        diff = new_length - current_best_length
        p = np.exp(-diff / temp)
        if new_length < current_best_length:
            return candidate, new_length
        elif np.random.rand() <= p:
            return candidate, new_length
        else:
            return current_best, current_best_length

    def run_sa(self):
        count = 0
        max_iter = self.max_iter
        best_path = self.current_path
        best_distance = self.compute_total_distance(self.current_path, self.distance_matrix)
        while self.T0 > self.Tf and count < max_iter:
            count += 1
            candidate_path = self.get_new_path(self.current_path.copy())
            self.current_path, current_length = self.eval_fire(best_path, candidate_path, self.T0)
            if current_length < best_distance:
                best_distance = current_length
                best_path = self.current_path
            self.T0 *= self.cooling_rate   
        return self.location[best_path], best_distance

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
city_names = ["Seoul", "Busan", "Incheon", "Daejeon", "Goyang", "Bucheon", "Daegu", "Jeju", "Yeosu", "Ulsan"]
city_coords = [
    [127.0276, 37.4979],  # Seoul
    [129.0756, 35.1796],  # Busan
    [126.7052, 37.4563],  # Incheon
    [127.3845, 36.3504],  # Daejeon
    [126.9780, 37.5665],  # Goyang
    [126.6156, 37.5670],  # Bucheon
    [128.5912, 35.8660],  # Daegu
    [126.5312, 33.4996],  # Jeju
    [127.5193, 34.9400],  # Yeosu
    [129.3114, 35.5384],  # Ulsan
]
data = np.array(city_coords)

Best_path, Best = SA(num_city=data.shape[0], data=data.copy()).run_sa()

Best_path = np.vstack([Best_path, Best_path[0]])
path_indices = [np.where((data == coord).all(axis=1))[0][0] for coord in Best_path]
named_path = [city_names[i] for i in path_indices]
print("Best total distance:", Best)
print("Best path:", " -> ".join(named_path))
