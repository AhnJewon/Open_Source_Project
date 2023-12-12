import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

stroke = pd.read_csv('./dataset/brain_stroke.csv', header=0)

# 데이터 셋의 상세 정보
print(stroke.info())
print('\n')
print(stroke.describe())
print('\n')

# age열의 정보를 나이대에 따라 새로 정리
def age(x):
    if(x<=25): return '<25'
    elif(25<x<=45): return '26~45'
    elif(45<x<=61): return '46~61'
    else: return '61<'

st_age = pd.read_csv('./dataset/brain_stroke.csv', header=0)
st_age.age = st_age.age.transform(age)

# 나이와 성별에 따라 그룹화
st_age = st_age.groupby(['age','gender'])

for key, group in st_age:
    print('* key :', key)
    print('* number :', len(group))
    print(group.head())
    print('\n') 

# 나이와 성별의 따른 통계 평균
avg_age = st_age.mean('numeric_only')
print(avg_age)
print('\n')

# 고혈압 유무에 따른 그룹화
st_hyptens = stroke.groupby('hypertension')
avg_hyptens = st_hyptens.mean('numeric_only')
print(avg_hyptens)
print('\n')

# 심장병 유무에 따른 그룹화
st_heart = stroke.groupby('heart_disease')
avg_heart = st_heart.mean('numeric_only')
print(avg_heart)
print('\n')

# 고혈압 유무, 심장병 유무에 따른 그룹화
st_hypt_hrt = stroke.groupby(['hypertension','heart_disease'])
avg_hypt_hrt = st_hypt_hrt.mean('numeric_only')
print(avg_hypt_hrt)
print('\n')

# 흡연 상태에 따른 그룹화
st_smoke = stroke.groupby('smoking_status')
avg_smoke = st_smoke.mean('numeric_only')
print(avg_smoke)
print('\n')

# 고혈압 유무, 흡연 상태에 따른 그룹화
st_smk_hypts = stroke.groupby(['smoking_status','hypertension'])
avg_smk_hypts = st_smk_hypts.mean('numeric_only')
print(avg_smk_hypts)
print('\n')

# 고혈압 유무, 뇌졸증 유무에 따른 그룹화
st_stk_hypts = stroke.groupby(['stroke','hypertension'])
avg_stk_hypts = st_stk_hypts.mean('numeric_only')
std_stk_hypts = st_stk_hypts.std(numeric_only=True)
print(avg_stk_hypts)
print('\n')
print(std_stk_hypts)
print('\n')

for key, group in st_stk_hypts:
    print('* key :', key)
    print('* number :', len(group))
    print(group.head())
    print('\n') 

# avg_glucose_level열의 정보를 당뇨병 기준에 따라 새로 분류
st_glucose = pd.read_csv('./dataset/brain_stroke.csv', header=0)

def glucose(x):
    if(x<100): return '<100'
    elif(100<=x<=125): return '100~125'
    else: return '125<'

st_glucose.avg_glucose_level = st_glucose.avg_glucose_level.transform(glucose)

# 평균 혈당에 따른 그룹화
st_glucose = st_glucose.groupby('avg_glucose_level')
avg_glucose = st_glucose.mean('numeric_only')
print(avg_glucose)
print('\n')

# 뇌졸증 유무에 따른 혈당 통계
std_glucose = stroke.groupby('stroke')
std_glucose = std_glucose.agg({'avg_glucose_level':['min','max','std']})
print(std_glucose)

# 시각화화를 위한 데이터 정리
stroke_grph = stroke[['age','hypertension','heart_disease','avg_glucose_level','bmi','smoking_status','stroke']]
stroke_grph = stroke_grph[~stroke_grph['smoking_status'].str.contains("Unknown", na=False, case=False)]

stroke_grph['smoking_status'].replace('never smoked','0',inplace=True)
stroke_grph['smoking_status'].replace('formerly smoked','1',inplace=True)
stroke_grph['smoking_status'].replace('smokes','1',inplace=True)

stroke_grph['smoking_status'] = stroke_grph['smoking_status'].astype('Int64')

# 히트맵 그래프 생성
plt.figure(figsize = (15, 10))
sns.heatmap(stroke_grph.corr(), annot = True, cmap = 'coolwarm')

# 그리드 그래프 생성
grid_strk_grph = sns.pairplot(data = stroke, hue = 'gender')
plt.show()
plt.close()

# 뇌졸증 유무에따른 그래프 생성
for i in stroke_grph.columns:
    plt.figure(figsize = (15,6))
    sns.countplot(x=stroke_grph[i], data = stroke_grph, hue = 'stroke' , palette = 'hls')
    plt.xticks(rotation = 90)
    plt.show()