---

title: 'Deep Learning - 딥러닝 이론 정리 3-1 (optimizer 정리)'

categories: ['Data Science', 'Deep Learning']

tags: 
- 딥러닝

use_math: true

toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"

---

## 옵티마이저 (Optimizer)
---
신경망 모델을 최적화 하는데 필요한 파라미터.
모델 별로 학습하는 데이터, 노드, 레이어가 모두 다르고, 궁극적으로 해결하고자 하는 문제도 다르기 때문에 그 모든 요건에 맞춰 적절한 옵티마이저를 통해 최적화 해야한다.

---
### GD(Gradient Descent)

가장 기본이 되는 최적화 기법으로 경사를 따라 내려가면서 가중치를 업데이트 한다.  
오차의 가중치에 대한 편미분값(가중치 기울기)을 찾아 최저점을 찾도록 하는 방법이다.  

---

### SGD(Stochastic Gradient Descent)

확률적 경사하강법으로 전체 데이터 중 1개의 데이터를 뽑아 오차를 확인한 뒤 이에 대한 최적화를 진행 하는 것.   

전체 데이터를 다 학습하지 않기 때문에 빠르지만, 성능개선이 힘든 단점이 있기 때문에 mini-batch를 이용해 weight를 수정하면서 최적화를 진행한다.   

다른 변형 옵티마이저들의 시조가 된다.  

$W_i = W_{i-1} - \eta \frac {\delta L}{\delta W}$   

---

### Momentum
SGD의 단점을 보완하고자 가중치 업데이트 시 관성(momentum)을 넣어 업데이트 되도록 한다. 이전 학습의 gradient와 본 학습의 gradient를 모두 고려한다.   

local minimum에 수렴하거나 미분값이 0이되어 이동하지 않는 SGD의 단점을 보완했으나, global minimum으로의 수렴이 매우 느릴 수 있고, 자칫 관성에 의해 최적 점을 가로질러 지나칠 수 있는 단점도 존재.  

보통 이전학습의 결과는 0.9 반영, 본 학습의 결과는 0.1로 두고 반영한다.  


$v_i = \alpha v_{i-1} - \eta \frac {\delta L}{\delta W} (\therefore \alpha=0.9, \eta=0.1)$  

$W_i = W_{i-1} +v_i$   


---
### Adagrad(Adapted Gradient)
가중치들에 개별 기준을 적용하여 업데이트 - 업데이트가 많이 일어나지 않은 가중치 위주로 큰 학습률 변화를 준다는 개념.  

매 학습별 가중치 기울기의 제곱합 G(t)를 계산해 업데이트 시의 학습률$\eta$(에타)에 루트를 씌워 나눔으로써 G(t)에 따른 학습률 조정을 진행한다. 또한 추가적으로 아주 작은 값 $\epsilon$(엡실론)을 넣어줌으로써 분모가 0이 되는 것을 방지한다.  

단점으로는 G(t)가 계속 증가하기 때문에 후엔 학습률 자체가 너무 적어져 나중에 0에 수렴하게 된다면 큰 변동이 없어질 수 있다.  

$G_i=G_{i-1}+\frac{\delta L}{\delta W}\odot\frac{\delta L}{\delta W}$    
$W_i=W_{i-1}-\frac{\eta}{\sqrt G_i+\epsilon}\frac{\delta L}{\delta W}$    

---

### RMSProp
Adagrad의 방식에서 단순히 계속해서 gradient를 제곱해 학습률을 구하는 것이 아니라, 지수가중평균이동(Exponential Moving Average)를 통해 최신 기울기에 무게를 두고 이를 학습률에 반영해 G(t)가 계속 커지는 것을 반영한다.   
최근의 가중치 기울기가 완만할 경우, 다시 학습률이 더 올라갈 수 있게끔 장치를 해주는 방법.  

$G_i=\beta *G_{i-1}+(1-\beta)*\frac{\delta L}{\delta W}\odot\frac{\delta L}{\delta W}$ $(\therefore \beta=지수 가중치)$  
$W_i=W_{i-1}-\frac{\eta}{\sqrt G_i+\epsilon}\frac{\delta L}{\delta W}$  


---

### Adam (Adapted Momentum Estimation)
Adagrad(또는 RMSProp)과 momentum의 장점으로 가장 널리 사용되는 옵티마이저이다.  

Momentum과 RMSProp의 0으로 초기화 될 경우 학습 초기에 0에 가깝게 지속되는 문제를 본 학습의 학습률을 분모에 넣어 줌으로써 해결했다. RMSProp의 지수평균이동을 통해 최신 기울기 및 0분모를 방지하기도 하여 가장 보편적으로 사용된다.  

**기존 momentum : **
$m1=\alpha m_0+\eta\frac{\delta L}{\delta W} (\therefore\alpha=0.9, \eta=0.1)$   
keras에서의 사용을 돕기 위해 $\alpha$와 $\eta$를 $\beta_1$로 바꾸어 상호 tradeoff 관계로 만들어본다.  
$m1=\beta_1 m_0 +(1-\beta_1)\frac{\delta L}{\delta W} (\therefore\beta_1=0.9)$  
$m_0$이  0일 때, = $0.1 * \frac {\delta L}{\delta W}$  


**Adam 내의 momentum_1차:**
$m1=\frac{\beta_1 m_0}{1-\beta_1} +\frac{(1-\beta_1)\frac{\delta L}{\delta W}}{1-\beta_1} (\therefore\beta_1=0.9권장)$  
$m_0$이  0일 때, = $1 * \frac {\delta L}{\delta W}$  

가 되므로 가장 처음으로 업데이트 되는 $m_0 = 0$이라는 초기값으로 인해 학습 초반의 효과가 떨어지는 것을 방지할 수 있게 된 것이다. 이와 같이 RMSProp의 G에도 이를 적용하여 식을 작성한다.  

**기존 RMSProp:**
$v_1=\beta_2 *v_0+(1-\beta_2)*\frac{\delta L}{\delta W}\odot\frac{\delta L}{\delta W}$ $(\therefore \beta_2=지수 가중치)$  

**Adam 내의 RMSProp, momentum_2차:**
$v_1=\frac{\beta_2 *v_0}{1-\beta_2}+\frac{(1-\beta_2)*\frac{\delta L}{\delta W}\odot\frac{\delta L}{\delta W}}{1-\beta_2}$ $(\therefore \beta_2=지수 가중치 , 0.999권장)$  

**ADAM 최종 수식** 
$W_i=W_{i-1}-\frac{\eta}{\sqrt v_{i-1}+\epsilon}m_{i-1}$  


---
