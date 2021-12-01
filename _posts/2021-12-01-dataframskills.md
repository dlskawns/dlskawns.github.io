---
title: "Dataframe 내 Column, row의 제거방법drop, iloc, loc 이용(pandas)"
categories: ['Data Science', "python"]
tags: 
- 전처리
- 데이터프레임
---



데이터를 분석하기에 앞서 전처리(PREPROCESSING)와 탐색적 데이터분석(EDA:Exploratory Data Analysis)을 진행하는데, 그 과정에 항상 필요한 열(columns)과 행(rows)의 조작이 필수적이다.

그러한 조작(Data Manipulation)과 특성공학(Feature Engineering)들을 위한 열과 행의 삭제 방법을 기재해보겠다.

## drop 함수를 이용한 삭제

열(columns) 의 삭제: 
```python
df.drop(columns = ['A'])  # 한 개 열을 삭제할 때.
df.drop(columns = ['A','B'])  #여러개 열을 삭제할 때.

# columns를 안쓰고 axis 설정을 통해 삭제도 가능하다
df.drop('A', axis = 1)   #한 개 열을 삭제할 때.
df.drop(['A', 'B'], axis = 1)   #여러개 열을 삭제할 때.
```
![](https://images.velog.io/images/dlskawns/post/54f1386a-1284-45cf-8322-703adea42ba2/image.png)

행(row)의 삭제:
```python
df.drop(0)  # index 중 0 행 삭제, axis의 default값이 axis = 0이기 때문에 안쳐도 된다.
df.drop([1,2]) # 여러 행 삭제 - 1, 2 행을 삭제했다. 0과 3행만 남음
```
![](https://images.velog.io/images/dlskawns/post/6aa5f879-3085-4cc4-87ea-038c65c30417/image.png)

## iloc, loc을 이용한 삭제
iloc을 이용 - index location으로 행 또는 열 index로 슬라이싱이 가능하다.
열(columns) 의 삭제: 
```python
df.iloc[:, 1:] # 0번째 column을 삭제하고 나머지만 남긴다.
df.iloc[:,2:] # 0번째, 1번째 column 삭제 뒤 나머지만 남기기.
```
![](https://images.velog.io/images/dlskawns/post/abe58d64-5f8d-4ed4-8492-13b65c848ff6/image.png)

행(rows) 의 삭제:
```python
df.iloc[1:] # 0번째 행 삭제하고 나머지 추출
df.iloc[1:3]  # 1번째 행부터 2번째 행까지만 출력하여 0행과 3행을 삭제한다.
df[1:]    # row는 일반 슬라이싱이 defalut로 설정되어있어, iloc 없이도 index 슬라이싱으로 바로 삭제 가능하다.
```
![](https://images.velog.io/images/dlskawns/post/d8f2feb3-fdaa-4675-92e9-0f4067d51800/image.png)

loc을 이용 - location 함수로 행 또는 열의 이름으로 선택함으로 삭제한다.
열(columns) 의 삭제: 
```python
df.loc[:,'B':]  # B열부터 끝까지 출력
df.loc[:,'A':'B'] # A열부터 B열까지 출력
```
![](https://images.velog.io/images/dlskawns/post/d25a84b2-4b86-40c6-bef8-81283a26cf7f/image.png)
행(rows) 의 삭제:
```python
df.loc[1:] # 1행부터 끝까지 출력
df.loc[2:3] # 2행부터 3행까지 출력
```
![](https://images.velog.io/images/dlskawns/post/a4edb960-4a6d-4a42-82ab-2fcc49f00da1/image.png)
여기서 주의 할 점은 loc함수는 iloc과 달리 슬라이싱시 마지막 선택 행 또는 열을 포함하는것이다. 
또한 loc과 iloc은 선택을 위한 함수로 이를 적절히 이용하면 원하는 행 또는 열을 선택해서 추출할 수 있고, 원하는대로 변형을 할 수 있기 때문에 유용하게 쓰인다. 이 점 참고해서 다음엔 다른 것에 적용해보도록 하겠다.
