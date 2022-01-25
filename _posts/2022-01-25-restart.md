---

title: 'Machine Learning - Target 분포 불균형 / unbalanced class의 문제 해결하기'

categories: ['Data Science', 'Machine Learning']

tags: 
- 머신러닝, 타겟분포

use_math: true

toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"

---


## 타겟 불균형 정의:  
문제 해결을 위한 Target의 분포가 한쪽으로 치우쳐 있어 원하는 Target 예측 성능이 떨어질 수 있는 상태   

### 분류 문제  
대개 이진분류 문제에서 target 한쪽이 70%이상 될 경우를 의미하며, 이런 경우 accuracy만으로 성능을 측정하면 recall(재현율) 또는 precision(정밀도)가 떨어져 원하는 예측이 불가능할 수 있다.  

### 회귀 문제  
Target의 평균이 전체 분포의 오른쪽으로 벗어나있거나(Positively skewed) 왼쪽으로 벗어난 경우(Negatively skewed) 예측 모델의 성능 개선이 어려워진다.  

### 예시:  

```python
import pandas as pd
# Telco 고객 데이터
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 타겟 선정
target = 'Churn'

# 타겟 분포 확인
df[target].value_counts(),
df[target].value_counts(normalize = True)
```
```
[output]

No     0.73463
Yes    0.26537
Name: Churn, dtype: float64
```

Telco사의 고객 데이터를 이용해서 Target을 회원 탈퇴여부인 'Churn'으로 두어 그 분포를 살폈봤을 때,  
NO: 73% / YES: 26% 로 불균형의 클래스임을 확인할 수 있다. 



## 분류 문제:  

### 가중치(Weight) 계산:  

모델 작성시 class_weight / scale_pos_weight로 가중치를 주어서 계산하도록 한다. 해당 가중치는 분류기마다 약간의 이름이 다르지만, 대개 두 가지 내에서 통용된다.  
보통 num_neg(구하고자 하지않는 타겟) / num_pos(구하고자하는 타겟)으로 가중치를 둔다 (예: pos: 30%, neg: 70% >> weight = 70/30  

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,classification_report, plot_confusion_matrix


# 랜덤 포레스트 기본모델로 작성
pipe = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(max_depth= 5, n_estimators= 100, random_state= 2, n_jobs=-1)
)

pipe.fit(X_train, y_train)

print('검증 정확도:',pipe.score(X_test, y_test))
y_test_pred = pipe.predict(X_test)
print('F1:',f1_score(y_test, y_test_pred, pos_label = 'Yes'))
print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(pipe, X_test, y_test);
```

![](https://images.velog.io/images/dlskawns/post/ec1fae30-76cb-4d39-8491-890dd498c275/image.png)

위는 class_weight 가중치를 두지 않고 진행했을 때의 결과이다.  
다음으로는 class_weight 가중치를 'balanced'로 두고 진행해보겠다.  

```python
# 랜덤 포레스트 class_weight = 'balanced'를 추가
pipe = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(max_depth= 5,
                           n_estimators= 100,
                           random_state= 2,
                           n_jobs=-1,
                           class_weight = 'balanced')
)

pipe.fit(X_train, y_train)

print('검증 정확도:',pipe.score(X_test, y_test))
y_test_pred = pipe.predict(X_test)
print('F1:',f1_score(y_test, y_test_pred, pos_label = 'Yes'))
print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(pipe, X_test, y_test);
```

![](https://images.velog.io/images/dlskawns/post/e13a9558-ab32-449c-8f3b-0141a7f479a2/image.png)

검증정확도(accuracy)가 조금 떨어졌지만 f1 스코어가 확연히 상승한 것을 볼 수 있다.  
class_weight가중치를 두지 않았을 때엔 'No'에 대한 것만 잘 맞춰서 정확도가 높았지만, 가중치를 조절함으로써 수가 적은 'Yes'도 잘 맞출 수 있게 균형잡힌 모델이 되었다고 볼 수 있다.  

### Under Sampling(과소표집):  
실제 데이터보다 적게 샘플링을 해서 학습 시 균등하게 학습할 수 있도록 하는 방법이다. 비율에 맞도록 데이터를 제거하는 방식으로 유용한 데이터도 삭제될 수 있는 단점을 가진다.  

```python
y_train.value_counts()
```
```
[output]

No     3084
Yes    1141
Name: Churn, dtype: int64
```

기본적인 분포 상태가 위와 같은 경우, 과소 표집을 하게되면 수가 적은 Yes의 수에 맞춰 Sampling하게 된다.  

```python
from imblearn.under_sampling import RandomUnderSampler

under = RandomUnderSampler
X_train_under, y_train_under = under(random_state = 2, sampling_strategy = 'majority').fit_resample(X_train, y_train)
y_train_over.value_counts()
```
```
[output]

Yes    1141
No     1141
Name: Churn, dtype: int64
```

위와 같이 'Yes'의 수에 맞춰 'No'의 수가 균등하게 조절 된 것을 볼 수 있다.  
parameter 중 sampling_strategy = 0.5로 50%만 가중치를 둘 수 있으니 필요에 맞게 사용하면 된다.  

이를 통해 예측을 진행하여 성능을 확인해본다.  
```python
# under sampling


pipe = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(random_state= 2)
)

pipe.fit(X_train_under, y_train_under)

print('train 검증 정확도:',pipe.score(X_train, y_train))
print('test 검증 정확도:',pipe.score(X_test, y_test))
y_test_pred = pipe.predict(X_test)
print('F1:',f1_score(y_test, y_test_pred, pos_label = 'Yes'))
print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(pipe, X_test, y_test);
```

![](https://images.velog.io/images/dlskawns/post/fe9e4bee-d19c-4d50-83d5-8de8b7eab3cb/image.png)

좀 더 극단적인 결과를 확인하기 위해 파라미터를 random_state 외 모두 제거하고, 결과에 train set에 대한 예측 성능도 추가해보았다.   
결과적으로 샘플 수도 적은 상태에서 under sampling을 함으로써 학습자체도 전혀 균형을 못이뤄서 f1 스코어가 0.005인 것을 알 수 있다.   
이런 경우엔 과소표집으로 샘플링을 해선 균형잡힌 성능 개선이 힘들다.  



### Over Sampling(과대표집):  
적은 분포의 타겟을 실제 데이터보다 많이 샘플링하여 타겟 분포를 균등하게 하는 방법이다. 무작위로 복제하거나 일정 기준을 두고 복제하여 데이터 수를 균등하게 맞춘다.   
그렇기 때문에 Train set에선 잘 나오지만, Test set에선 성능이 저하되는 과적합의 문제를 갖고있고, 이를 해소하기 위해 SMOTE(Synthetic Minority Over-sampling Technique)를 이용하기도 한다.  

```python
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler
X_train_over , y_train_over = over(random_state =2, sampling_strategy='minority').fit_resample(X_train, y_train)
y_train_over.value_counts()
```
```
[output]

No     3084
Yes    3084
Name: Churn, dtype: int64
```

위와 같이 'No'의 수에 맞게 'Yes'가 무작위로 copy된 것을 볼 수 있다.  
이번에는 over sampling이 어느정도의 성능 개선을 보이는지 알아보도록 한다.  

```python
# over sampling

pipe = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(random_state= 2)
)

pipe.fit(X_train_over, y_train_over)

print('train 검증 정확도:',pipe.score(X_train, y_train))
print('test 검증 정확도:',pipe.score(X_test, y_test))
y_test_pred = pipe.predict(X_test)
print('F1:',f1_score(y_test, y_test_pred, pos_label = 'Yes'))
print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(pipe, X_test, y_test);
```

![](https://images.velog.io/images/dlskawns/post/2b050a4f-1e43-4176-834e-e1efc43b8059/image.png)

과소표집과 마찬가지로 train set에 대한 정확도를 보면 과적합의 문제가 분명 없지 않은듯 하다. 또한 accuracy와 f1 스코어도 맨 처음 모델에 비해 떨어진 것을 볼 수 있다.  
이번에는 과대표집의 개선된 버젼인 SMOTE를 진행 해보겠다. 

#### SMOTE  
분포가 적은 클래스에서 샘플 별로 KNN(K-Nearest-Neighbors)를 찾아 선을 그어 무작위 점을 생성하는 방법이다.   
```python
from imblearn.over_sampling import SMOTE
from category_encoders import OrdinalEncoder

enc = OrdinalEncoder()
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.fit_transform(X_test)

X_train_smoted, y_train_smoted = SMOTE(random_state= 2, sampling_strategy='minority').fit_resample(X_train_enc, y_train)
y_train_smoted.value_counts()
```
```
[output]

No     3084
Yes    3084
Name: Churn, dtype: int64
```
SMOTE는 Nearest Neighbors를 찾는 과정에서 수치화가 필요하기 때문에 Ordinal Encoder를 이용해 X_train을 변경해주었고, 샘플 수는 over sampling이므로 많은 쪽이었던 'No'의 수로 통일 되었다.  
```python
# SMOTE

pipe = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(random_state= 2)
)

pipe.fit(X_train_smoted, y_train_smoted)

print('train 검증 정확도:',pipe.score(X_train_enc, y_train))
print('test 검증 정확도:',pipe.score(X_test_enc, y_test))
y_test_pred = pipe.predict(X_test_enc)
print('F1:',f1_score(y_test, y_test_pred, pos_label = 'Yes'))
print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(pipe, X_test_enc, y_test);
```
![](https://images.velog.io/images/dlskawns/post/a5813ef4-824a-43a7-bfdc-d7adda566597/image.png)

최종적인 결과를 보았을 때 over sampling을 진행할 경우, SMOTE를 사용하면 조금 더 안정된 성능을 보여주는 것을 알 수 있다. 각 방법들을 Data set에 맞는 적절한 라이브러리를 이용하면 된다.  

### 회귀 문제:  
회귀문제에 대한 모델도 정규분포에서 더 나은 성능을 보이기 때문에 가능한 맞춰주는것이 중요하다.  

#### 1. 이상치 제거:  


#### 2-1 LOG TRANSFORM:

numpy의 log1p method를 사용해서 비대칭 분포를 정규분포로 변환한다. -> $\ln(1+x)$  
값이 변화되어 해석에 문제 발생시 역함수 진행 -> $\exp(x)$-1  

#### 2-2 TransformedTargetRegressor 사용:

TransformedTargetRegressor를 통해 pipeline과 연계하여 code를 단순화 할 수 있다.  
