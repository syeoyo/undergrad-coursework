import numpy as np
import cvxpy as cp

max_w_cost = 30000

w_size_capacity = np.array([3000, 5000, 10000])
w_cost = np.array(
    [
        [10000, 15000, 20000],
        [8000, 12000, 17000],
    ]
)
demand = np.array([2000, 2500, 1000, 3500])

c_fw = np.array([10, 15])
c_wr = np.array(
    [
        [2, 3, 5, 4],
        [5, 4, 6, 7],
    ]
)

# 변수 정의
y_ws = cp.Variable((2, len(w_size_capacity)), boolean=True)
x_fw = cp.Variable(2, integer=True)
x_wr = cp.Variable((2, len(demand)), integer=True)

# 제약조건
constraints = []

# 각 창고는 최대 1개의 크기만 선택 가능
for w in range(2):
    constraints.append(cp.sum(y_ws[w, :]) <= 1)

# 창고 설치 비용 제약
constraints.append(cp.sum(cp.multiply(w_cost, y_ws)) <= max_w_cost)

# 창고 용량 제약
for w in range(2):
    constraints.append(x_fw[w] <= w_size_capacity @ y_ws[w, :])
    constraints.append(x_fw[w] >= 0)

# 창고 입출고량 균형 제약
for w in range(2):
    constraints.append(x_fw[w] == cp.sum(x_wr[w, :]))

# 소매점 수요 충족 제약
for r in range(len(demand)):
    constraints.append(cp.sum(x_wr[:, r]) >= demand[r])
    
# 음수 방지 제약
constraints.append(x_wr >= 0)

# 목적함수
objective = (cp.sum(cp.multiply(c_fw, x_fw)) + 
            cp.sum(cp.multiply(c_wr, x_wr)) +
            cp.sum(cp.multiply(w_cost, y_ws)))

# 문제 정의 및 풀이
prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve()

print(f"Objective value: {prob.value}")

print("\n창고 설치 위치 및 용량:")
for w in range(2):
    print(f"입지 {w+1}:")
    size_found = False
    for i, val in enumerate(y_ws[w,:].value):
        if val == 1:
            print(f"최적 건설 크기: {w_size_capacity[i]}")
            size_found = True
    if not size_found:
        print("최적 건설 크기: None")

print("\n물류 이송 경로(공장 ->창고):")
for w in range(2):
    if x_fw[w].value > 0:
        print(f"공장 -> 창고{w+1}: {x_fw[w].value:.0f}")
    else:
        print(f"공장 -> 창고{w+1}: 0")

print("\n물류 이송 경로(창고->소매점):")
for w in range(2):
    for r in range(len(demand)):
        if x_wr[w, r].value > 0:
            print(f"창고{w+1} -> 소매점{r+1}: {x_wr[w, r].value:.0f}")
        else:
            print(f"창고{w+1} -> 소매점{r+1}: 0")