# Traveling Salesman Problem using Genetic Algorithm

from itertools import permutations
from random import shuffle
import random
import numpy as np

def initial_population(cities, n_population = 250):
    population_perm = []
    all_possible_perm = list(permutations(cities))
    random_idx = random.sample(range(0,len(all_possible_perm)),n_population)
    for i in random_idx:
        population_perm.append(list(all_possible_perm[i]))  
    return population_perm

def euclidean_distance(city_1, city_2):
    coord_1 = coordinates[city_1]
    coord_2 = coordinates[city_2]
    return np.sqrt(np.sum((np.array(coord_1) - np.array(coord_2))**2))

def individual_total_distance(individual_perm):
    total_dist = 0
    for i in range(0, len(individual_perm)):
        if(i == len(individual_perm) - 1):
            total_dist += euclidean_distance(individual_perm[i], individual_perm[0])
        else:
            total_dist += euclidean_distance(individual_perm[i], individual_perm[i+1])
    return total_dist

def fitness_prob(population):
    all_total_dist = []
    for i in range (0, len(population)):
        all_total_dist.append(individual_total_distance(population[i]))
    population_fitness = max(all_total_dist) - np.array(all_total_dist)
    fitness_sum = population_fitness.sum()
    if fitness_sum == 0:
        return np.ones(len(population)) / len(population)
    return population_fitness / fitness_sum

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_index = np.searchsorted(population_fitness_probs_cumsum, np.random.uniform(0, 1))
    return population[selected_index]

def crossover(p1, p2):
    cut = round(random.uniform(1, len(cities_names) - 1))
    o1 = []
    o2 = []
    o1 = p1 [0:cut]
    o1 += [city for city in p2 if city not in o1]
    o2 = p2 [0:cut]
    o2 += [city for city in p1 if city not in o2]
    return o1, o2

def mutation(offspring):
    idx_1 = random.randint(0, len(cities_names) - 1)
    idx_2 = random.randint(0, len(cities_names) - 1)
    temp = offspring [idx_1]
    offspring[idx_1] = offspring[idx_2]
    offspring[idx_2] = temp
    return(offspring)

def generate_next_generation(population, crossover_per, mutation_per, n_population):
    fitness_probs = fitness_prob(population)
    parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

    offsprings = []
    for i in range(0, len(parents), 2):
        o1, o2 = crossover(parents[i], parents[i+1])
        if random.random() > (1 - mutation_per):
            o1 = mutation(o1)
        if random.random() > (1 - mutation_per):
            o2 = mutation(o2)
        offsprings.extend([o1, o2])

    mix_offspring = parents + offsprings
    fitness_probs = fitness_prob(mix_offspring)
    sorted_idx = np.argsort(fitness_probs)[::-1]
    best_idx = sorted_idx[0:n_population]
    next_generation = [mix_offspring[i] for i in best_idx]
    return next_generation

def run_ga(cities, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities, n_population)
    for i in range(n_generations):
        population = generate_next_generation(population, crossover_per, mutation_per, n_population)
    return population

n_population = 300
crossover_per = 0.8
mutation_per = 0.05
n_generations = 200

x = [126, 129, 127, 128, 126, 128, 127, 126, 126, 127]
y = [37, 35, 37, 35, 35, 36, 35, 37, 35, 35]
cities_names = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Ulsan", "Suwon", "Goyang", "Changwon", "Yongin"]
coordinates = dict(zip(cities_names, zip(x, y)))

best_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

potential_total_distances = []
for i in range(0, n_population):
    potential_total_distances.append(individual_total_distance(best_offspring[i]))

index_min = np.argmin(potential_total_distances)
min_distance = min(potential_total_distances)
shortest_path = best_offspring[index_min]

print("\n===== Genetic Algorithm Parameters =====")
print(f"Population Size (n_population): {n_population}")
print(f"Crossover Rate (crossover_per): {crossover_per}")
print(f"Mutation Rate (mutation_per): {mutation_per}")
print(f"Number of Generations (n_generations): {n_generations}")
print(f"Encoding Method (Chromosome): Permutation-based Encoding")
print(f"Fitness Function: Minimization of total travel distance between cities")

print("\n===== Shortest Path Found =====")
print(" -> ".join(shortest_path))
print(f"Total Distance: {min_distance:.3f}")