from gurobipy import Model, GRB

model = Model()

center = range(3)
city = range(4)

# Parameters
cap = [30, 55, 40]
fc = [20_000, 30_000, 27_500]
tc = [
    [600, 500, 900, 300],
    [400, 300, 500, 600],
    [500, 800, 200, 400],
]
d = [11, 18, 15, 25]

# Variables
x = model.addVars(center, city, vtype=GRB.INTEGER, name="x")
y = model.addVars(center, vtype=GRB.BINARY, name="y")

# Objective
model.setObjective(
    sum(fc[i] * y[i] for i in center)
    + sum(tc[i][j] * x[i, j] for i in center for j in city),
    GRB.MINIMIZE,
)

# Constraints
for j in city:
    model.addConstr(sum(x[i, j] for i in center) >= d[j])

for i in center:
    model.addConstr(sum(x[i, j] for j in city) <= cap[i] * y[i])

model.optimize()

print(f"Total cost: ${model.objVal:,.2f}")
print()

for i in center:
    if y[i].x == 1:
        print(f"Center {i+1}: Open")
    else:
        print(f"Center {i+1}: Closed")
print()

for i in center:
    for j in city:
        if x[i, j].x > 0:
            print(f"Center {i+1} -> City {j+1}: {int(x[i,j].x)} truckloads")
