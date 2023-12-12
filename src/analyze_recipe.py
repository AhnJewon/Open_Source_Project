import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import cluster
from sklearn.metrics import silhouette_samples

df = pd.read_csv('./dataset/All_Diets.csv', header=0)

# 데이터 상세 정보 확인
print(df.head())
print('\n')

print(df.info())
print('\n')

# 데이터 null 값존재 확인
print(df.isnull().any())
print('\n')

# Diet_type으로 데이터 그룹화
df_type = df.groupby('Diet_type')

for key, group in df_type:
    print('* key :', key)
    print('* number :', len(group))
    print(group.head())
    print('\n') 

# Diet_type 그룹 통계 평균
df_dash = df_type.mean('numeric_only')

print(df_dash)
print('\n')

# dash 식이요법의 통계평균
print(df_dash.loc['dash'])
print('\n')

# vegan 식이요법의 통계평균
print(df_dash.loc['vegan'])
print('\n')

# paleo 식이요법의 통계평균
print(df_dash.loc['paleo'])
print('\n')

# mediterranean 식이요법의 통계평균
print(df_dash.loc['mediterranean'])
print('\n')

# keto 식이요법의 통계평균
print(df_dash.loc['keto'])
print('\n')

# 데이터 군집화
ndf = df[['Diet_type','Protein(g)','Carbs(g)','Fat(g)']]

onehot_type = pd.get_dummies(ndf['Diet_type'],dtype='int')
ndf = pd.concat([ndf,onehot_type], axis=1)

ndf.drop(['Diet_type'],axis=1,inplace=True)
print(ndf.head())

X = preprocessing.StandardScaler().fit(ndf).transform(ndf)

print(X[:5])

kmeans = cluster.KMeans(init='k-means++',n_clusters=5,n_init=10)

kmeans.fit(X)

cluster_label = kmeans.labels_
print(cluster_label)
print('\n')

ndf['Cluster'] = cluster_label
print(ndf.head())

# 군집화 그래프 생성
ndf.plot(kind='scatter', x='Protein(g)', y='Carbs(g)', c='Cluster',
         colormap='Set1',colorbar=False,figsize=(10,10))
ndf.plot(kind='scatter', x='Carbs(g)', y='Fat(g)', c='Cluster',
         colormap='Set1',colorbar=False,figsize=(10,10))
ndf.plot(kind='scatter', x='Fat(g)', y='Protein(g)', c='Cluster',
         colormap='Set1',colorbar=True,figsize=(10,10))

plt.show()
plt.close()

# 모델 평가 검증
labels = kmeans.labels_
print(silhouette_samples(X,labels,metric='euclidean'))