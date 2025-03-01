import numpy as np
import matplotlib.pyplot as plt

# 1-a
# 표준정규분포 N(0,1)에서 100차원의 샘플 500개 생성
samples_01 = np.random.normal(loc=0, scale=1, size=(500, 100))

# 1-b
# 첫 번째 차원(feature) 선택
first_feature = samples_01[:, 0]

# 표준정규분포의 이론적 확률밀도함수 생성
x = np.linspace(min(first_feature), max(first_feature), 100)
pdf = 1/(np.sqrt(2*np.pi)) * np.exp(-x**2/2)

# 히스토그램을 확률밀도로 정규화하여 다시 그리기
plt.figure(figsize=(8, 6))
plt.hist(first_feature, bins=30, density=True, alpha=0.7, edgecolor='black', label='Sample Distribution')
plt.plot(x, pdf, 'r-', lw=2, label='Standard Normal PDF')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title("Sample Distribution of First Feature vs Standard Normal Distribution")
plt.legend()
plt.grid(True)
plt.show()

# 1-c
# N(2,5)에서 500개의 샘플 생성
samples_25 = np.random.normal(loc=2, scale=5, size=(500, 100))

# 가중합 계산 (N(0,1)의 70% + N(2,5)의 30%)
weighted_samples = 0.7 * samples_01 + 0.3 * samples_25

# 하나의 샘플에 대한 예시 출력
print("\n단일 샘플의 가중합 예시:")
print("N(0,1)의 샘플:", samples_01[0,0])
print("N(2,5)의 샘플:", samples_25[0,0]) 
print("가중합 결과:", weighted_samples[0,0])
print("계산 검증: 0.7 *", samples_01[0,0], "+ 0.3 *", samples_25[0,0], "=", 0.7*samples_01[0,0] + 0.3*samples_25[0,0])


# 1-d
# 첫 번째 차원의 데이터 추출
weighted_first_feature = weighted_samples[:, 0]

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(weighted_first_feature, bins=30, density=True, alpha=0.7, edgecolor='black', label='Weighted Sample Distribution')

# N(0,1)과 N(2,5)의 가중합 이론적 곡선 추가
x = np.linspace(min(weighted_first_feature), max(weighted_first_feature), 100)
pdf_01 = 1/(np.sqrt(2*np.pi)) * np.exp(-(x**2)/2)  # N(0,1)의 PDF
pdf_25 = 1/(np.sqrt(2*np.pi*25)) * np.exp(-(x-2)**2/(2*25))  # N(2,5)의 PDF
weighted_pdf = 0.7 * pdf_01 + 0.3 * pdf_25  # 7:3 가중합
plt.plot(x, weighted_pdf, 'r-', lw=2, label='Weighted Theoretical PDF (0.7×N(0,1) + 0.3×N(2,5))')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title("Weighted Sample Distribution (70% N(0,1) + 30% N(2,5)) vs Standard Normal")
plt.legend()
plt.grid(True)
plt.show()




