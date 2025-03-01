import numpy as np
import cvxpy as cp

demand = np.array([250, 350, 100, 200, 400, 300])

distance_matrix = np.array(
    [
        [0, 6, 2, 5, 4, 4],
        [6, 0, 6, 2, 9, 8],
        [2, 6, 0, 4, 3, 2],
        [5, 2, 4, 0, 7, 6],
        [4, 9, 3, 7, 0, 5],
        [4, 8, 2, 6, 5, 0],
    ]
)

binary_distance_matrix = np.array(
    [
        [1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 1],
    ]
)

nodes = ["A", "B", "C", "D", "E", "F"]

# Original problem
x_i = cp.Variable(len(nodes), boolean=True)
y_i = cp.Variable(len(nodes), boolean=True)

constraints = []

for index, binary_distance in enumerate(binary_distance_matrix):
    constraints.append(binary_distance @ y_i >= x_i[index])

constraints.append(cp.sum(y_i) <= 1)

objective = cp.Maximize(demand @ x_i)

problem = cp.Problem(objective, constraints)
problem.solve()

print("Before column elimination")
print(f"\nObjective value: {problem.value}")

print("x_i values:")
for i, val in enumerate(x_i.value):
    print(f"x_i[{i}]: {val}")
print("\ny_i values:")
for i, val in enumerate(y_i.value):
    print(f"y_i[{i}]: {val}")

# Column elimination problem
column_elimination_matrix = np.array(
    [
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
    ]
)

x_i = cp.Variable(len(nodes), boolean=True)
y_i = cp.Variable(len(nodes), boolean=True)

constraints = []

for index, binary_distance in enumerate(column_elimination_matrix):
    constraints.append(binary_distance @ y_i >= x_i[index])

constraints.append(cp.sum(y_i) <= 1)

objective = cp.Maximize(demand @ x_i)

problem = cp.Problem(objective, constraints)
problem.solve()

print("\nAfter column elimination")
print(f"\nObjective value: {problem.value}")

print("x_i values:")
for i, val in enumerate(x_i.value):
    print(f"x_i[{i}]: {val}")
print("\ny_i values:")
for i, val in enumerate(y_i.value):
    print(f"y_i[{i}]: {val}")
