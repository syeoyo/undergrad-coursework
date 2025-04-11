import pandas as pd
import matplotlib.pyplot as plt

# Subgroup data 입력
xbar_data = [8.3, 8.1, 7.9, 6.3, 8.5, 7.5, 8.0, 7.4, 6.4, 7.5,
             8.8, 9.1, 5.9, 9.0, 6.4, 7.3, 5.3, 7.6, 8.1, 8.0]
R_data =    [2,   3,   1,   5,   3,   4,   3,   2,   2,   4,
             3,   5,   3,   6,   3,   3,   2,   4,   3,   2]

df = pd.DataFrame({'Subgroup': list(range(1, 21)),
                   'x̄': xbar_data,
                   'R': R_data})

# 통계값 계산
xbar_bar = df['x̄'].mean()
R_bar = df['R'].mean()

# 관리도 상수 (n = 2)
A2 = 1.880
D3 = 0.000
D4 = 3.267

# x̄ 관리한계
UCL_x = xbar_bar + A2 * R_bar
LCL_x = xbar_bar - A2 * R_bar

# R 관리한계
UCL_R = D4 * R_bar
LCL_R = D3 * R_bar

# ───────────────────────────────────────
# x̄-chart 그리기
plt.figure(figsize=(12, 5))
plt.plot(df['Subgroup'], df['x̄'], marker='o', label='x̄', color='#0069E7')
plt.axhline(y=xbar_bar, color='#656565', linestyle='-', label='Center Line')
plt.axhline(y=UCL_x, color='#656565', linestyle='--', label='UCL')
plt.axhline(y=LCL_x, color='#656565', linestyle='--', label='LCL')
plt.xticks(df['Subgroup'])  # ⬅️ x축 눈금 설정
plt.title('x̄ Chart')
plt.xlabel('Subgroup')
plt.ylabel('x̄')
plt.legend()
# plt.grid(True)
plt.show()

# R-chart 그리기
plt.figure(figsize=(12, 5))
plt.plot(df['Subgroup'], df['R'], marker='o', label='R', color='#009B31')
plt.axhline(y=R_bar, color='#656565', linestyle='-', label='Center Line')
plt.axhline(y=UCL_R, color='#656565', linestyle='--', label='UCL')
plt.axhline(y=LCL_R, color='#656565', linestyle='--', label='LCL')
plt.xticks(df['Subgroup'])  # ⬅️ x축 눈금 설정
plt.title('R Chart')
plt.xlabel('Subgroup')
plt.ylabel('R')
plt.legend(loc='upper right')
# plt.grid(True)
plt.show()