# Traveling Salesman Problem using Particle Swarm Optimization 

import random
import math

class Particle:
    
    def __init__(self, path, cost = None):
        self.path = path
        self.pbest = path
        self.current_cost = cost if cost else self.path_cost(self.path)
        self.pbest_cost = cost if cost else self.path_cost(self.path)
        self.velocity = []

    def reset_velocity(self):
        self.velocity.clear()

    def update_costs_and_pbest(self):
        self.current_cost = self.path_cost(self.path)
        if self.current_cost < self.pbest_cost:
            self.pbest = self.path
            self.pbest_cost = self.current_cost

    def path_cost(self, path):
        return sum([math.hypot(path[i][1][0] - path[i - 1][1][0], path[i][1][1] - path[i - 1][1][1]) for i in range(len(path))])


class PSO:
    
    def __init__(self, iterations, population_size, gbest_probability=1.0, pbest_probability=1.0, cities = None):
        self.cities = cities
        self.gbest = None
        self.gcost_iter = []
        self.iterations = iterations
        self.population_size = population_size
        self.particles = []
        self.gbest_probability = gbest_probability
        self.pbest_probability = pbest_probability
        solutions = self.initial_population()
        self.particles = [Particle(path=solution) for solution in solutions]

    def random_init(self):
        return random.sample(self.cities, len(self.cities))

    def greedy_init(self, start_index):
        unvisited = self.cities[:]
        del unvisited[start_index]
        route = [self.cities[start_index]]
        while len(unvisited):
            index, nearest_city = min(enumerate(unvisited), key=lambda item: math.hypot(item[1][1][0] - route[-1][1][0], item[1][1][1] - route[-1][1][1]))
            route.append(nearest_city)
            del unvisited[index]
        return route
    
    def initial_population(self):
        random_population = [self.random_init() for _ in range(self.population_size - 1)]
        greedy_population = [self.greedy_init(0)]
        return [*random_population, *greedy_population]

    def run(self):
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
        for _ in range(self.iterations):
            self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
            self.gcost_iter.append(self.gbest.pbest_cost)

            for particle in self.particles:
                particle.reset_velocity()
                temp_velocity = []
                gbest = self.gbest.pbest[:]
                new_path = particle.path[:]

                for i in range(len(self.cities)):
                    if new_path[i] != particle.pbest[i]:
                        swap = (i, particle.pbest.index(new_path[i]), self.pbest_probability)
                        temp_velocity.append(swap)
                        new_path[swap[0]], new_path[swap[1]] = new_path[swap[1]], new_path[swap[0]]

                for i in range(len(self.cities)):
                    if new_path[i] != gbest[i]:
                        swap = (i, gbest.index(new_path[i]), self.gbest_probability)
                        temp_velocity.append(swap)
                        gbest[swap[0]], gbest[swap[1]] = gbest[swap[1]], gbest[swap[0]]

                particle.velocity = temp_velocity

                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_path[swap[0]], new_path[swap[1]] = new_path[swap[1]], new_path[swap[0]]

                particle.path = new_path
                particle.update_costs_and_pbest()
        
        print(f"Shortest Distance is {self.gbest.pbest_cost}")
        print(" -> ".join([city[0] for city in pso.gbest.pbest]))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if __name__ == "__main__":
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
    cities = list(zip(city_names, city_coords))
    pso = PSO(iterations=2000, population_size=500, pbest_probability=0.7, gbest_probability=0.3, cities=cities)
    pso.run()