#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import researchpy as rp
from scipy import stats
from scipy.stats import shapiro,bartlett
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# # 0. 데이터 로드 및 기본 통계량 출력

# In[29]:


data = pd.read_csv('diamond.csv')


# In[34]:


data


# In[4]:


data.describe()


# In[37]:


rp.summary_cont(data.carat)


# In[7]:


rp.summary_cont(data.price)


# # 1. 정규성 검정

# In[41]:


# 서브플롯 생성
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 첫 번째 Q-Q plot
pg.qqplot(data.carat[data.cut=="Good"], ax=axs[0])
axs[0].set_title('Good_carat')

# 두 번째 Q-Q plot
pg.qqplot(data.carat[data.cut=="Ideal"], ax=axs[1])
axs[1].set_title('Ideal_carat')

# 세 번째 Q-Q plot
pg.qqplot(data.carat[data.cut=="Premium"], ax=axs[2])
axs[2].set_title('Premium_carat')

# 그래프 간격 조정
plt.tight_layout()

# 그래프 표시
plt.show()


# In[31]:


#정규성 확인 - 집단(수준)별로 실시
# 귀무가설: 정규분포를 따른다
# 대립가설: 정규분포를 따르지 않는다.

from  scipy.stats import shapiro
print(shapiro(data.carat[data.cut=="Good"]))
print(shapiro(data.carat[data.cut=="Ideal"]))
print(shapiro(data.carat[data.cut=="Premium"]))


# In[42]:


# 서브플롯 생성
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 첫 번째 Q-Q plot
pg.qqplot(data.price[data.cut=="Good"], ax=axs[0])
axs[0].set_title('Good_price')

# 두 번째 Q-Q plot
pg.qqplot(data.price[data.cut=="Ideal"], ax=axs[1])
axs[1].set_title('Ideal_price')

# 세 번째 Q-Q plot
pg.qqplot(data.price[data.cut=="Premium"], ax=axs[2])
axs[2].set_title('Premium_price')

# 그래프 간격 조정
plt.tight_layout()

# 그래프 표시
plt.show()


# In[32]:


#정규성 확인 - 집단(수준)별로 실시
# 귀무가설: 정규분포를 따른다
# 대립가설: 정규분포를 따르지 않는다.

from  scipy.stats import shapiro
print(shapiro(data.price[data.cut=="Good"]))
print(shapiro(data.price[data.cut=="Ideal"]))
print(shapiro(data.price[data.cut=="Premium"]))


# # 2. 등분산 검정

# In[43]:


#등분산성 확인 - 바틀렛 검증
# 귀무가설: 등분산이다.
# 대립가설: 이분산이다.

from scipy.stats import bartlett
print(bartlett(data.carat[data.cut=="Good"],
      data.carat[data.cut=="Ideal"],
      data.carat[data.cut=="Premium"]))


# In[44]:


#등분산성 확인 - 바틀렛 검증
# 귀무가설: 등분산이다.
# 대립가설: 이분산이다.

from scipy.stats import bartlett
print(bartlett(data.price[data.cut=="Good"],
      data.price[data.cut=="Ideal"],
      data.price[data.cut=="Premium"]))


# # 3. One-way ANOVA

# In[45]:


model = ols('carat ~ C(cut)', data).fit()
anova_lm(model)


# In[46]:


model = ols('price ~ C(cut)', data).fit()
anova_lm(model)


# # 4. 사후검정

# In[21]:


from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats

comp_carat = MultiComparison(data.carat, data.cut)


# In[22]:


#BONFERRONI

result, a1, a2 = comp_carat.allpairtest(scipy.stats.ttest_ind, method='bonf')
result


# In[50]:


#Tuckey's test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
hsd = pairwise_tukeyhsd(data['carat'], data['cut'], alpha=0.05)
hsd.summary()


# In[49]:


comp_price = MultiComparison(data.price, data.cut)


# In[26]:


#BONFERRONI

result, a1, a2 = comp_price.allpairtest(scipy.stats.ttest_ind, method='bonf')
result


# In[51]:


#Tuckey's test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
hsd = pairwise_tukeyhsd(data['price'], data['cut'], alpha=0.05)
hsd.summary()

