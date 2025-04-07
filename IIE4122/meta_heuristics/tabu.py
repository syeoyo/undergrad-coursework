# Traveling Salesman Problem using Tabu Search

import numpy as np
from collections import deque

coordinates = np.array([
    [0, 0],    # city 1
    [5, 2],    # city 2
    [10, 0],   # city 3
    [10, 10],  # city 4
    [0, 10],   # city 5
])

def calculate_euc_distance(coords):
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = np.sqrt(
                (coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2
            )
    return distance_matrix

def calculate_total_distance(solution, distance_matrix):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[solution[i]][solution[i + 1]]
    total_distance += distance_matrix[solution[-1]][solution[0]]
    return total_distance

def tabu_search(distance_matrix, n_iter=100, tabu_size=5):
    n = len(distance_matrix)

    current_best_solution = list(range(n))
    current_best_distance = calculate_total_distance(current_best_solution, distance_matrix)

    best_solution = current_best_solution.copy()
    best_distance = current_best_distance

    tabu_list = deque(maxlen=tabu_size)

    for _ in range(n_iter):
        neighborhood = []
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = current_best_solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                if neighbor not in tabu_list:
                    neighborhood.append(
                        (neighbor, calculate_total_distance(neighbor, distance_matrix))
                    )

        neighborhood.sort(key=lambda x: x[1])
        candidate, candidate_distance = neighborhood[0]

        current_best_solution = candidate
        current_best_distance = candidate_distance

        tabu_list.append(current_best_solution)

        if current_best_distance < best_distance:
            best_solution = current_best_solution
            best_distance = current_best_distance

    return best_solution, best_distance


distance_matrix = calculate_euc_distance(coordinates)
best_solution, best_distance = tabu_search(distance_matrix)

best_route = [f"city {i+1}" for i in best_solution] + [f"city {best_solution[0]+1}"]

print("Best route:", best_route)
print("Total distance:", best_distance)