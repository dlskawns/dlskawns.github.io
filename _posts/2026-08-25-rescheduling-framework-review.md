---
title: 'Paper Review - 설비가 멈춰 스케줄을 다시 짜려다 보니, 그 문제엔 이미 이름이 있었다 (feat. 재스케줄링 프레임워크, nervousness)'

date: 2026-08-25 11:00:00 +0900

categories: ["paper-review"]

tags:
  - "재스케줄링"
  - "스케줄링"
  - "제조"
  - "논문리뷰"

use_math: true

toc: true
toc_label: "목차"
toc_icon: "list"
---
[지난 글](/optimization/mes-basics/)에서 MES를 정리하면서 마지막에 이렇게 적었다. 설비 한 대가 멈추면 WIP가 어떻게 밀리는지 재보겠다고.  
그 다음 질문이 자연스럽게 따라왔다. **밀린 걸 알았으면 스케줄을 다시 짜야 하는데, 얼마나 바꿔야 하나.**  

전부 다시 짜면 최적이지만 현장은 이미 그 계획대로 움직이고 있다.  
작업자는 다음 순서를 알고 있고 자재는 그 순서대로 대기 중이다. 다 뒤집으면 그 자체가 비용이다.  

근무표에서 하던 걸 그대로 옮기면 되겠다고 생각하고 자료를 뒤졌는데, 찾아보니 **이 문제엔 이미 이름이 붙어 있었고 분류 체계까지 확립돼 있었다.**  
그래서 이 글은 그 분류 체계를 정리하고, 내가 만들려는 것이 어느 칸에 들어가는지 확인하는 기록이다.  

측정은 없다. 대신 원문 PDF를 직접 받아 개념 정의를 하나씩 대조했다.  

## 읽은 논문

> Rescheduling manufacturing systems: a framework of strategies, policies, and methods  
> Guilherme E. Vieira, Jeffrey W. Herrmann, Edward Lin  
> [원문 PDF (University of Maryland ISR)](https://isr.umd.edu/Labs/CIM/projects/jos-rescheduling.pdf)  

서베이 논문이다. 새 알고리즘을 제안하는 게 아니라, 흩어져 있던 재스케줄링 연구를 하나의 분류 체계로 묶는다.  

논문이 밝히는 집필 동기가 인상적이었다.  

> 많은 연구가 재스케줄링을 다루지만, 그 전략과 정책과 방법에 대한 **표준 정의나 분류가 없다**.  

내가 겪은 게 정확히 이거였다.  
같은 걸 어떤 글은 dynamic scheduling이라 부르고 어떤 글은 reactive scheduling이라 불러서, 서로 다른 얘기인지 같은 얘기인지 알 수가 없었다.  

<br>
<br>
<br>

## 문제에 이미 이름이 있었다

### nervousness — 잦은 변경 자체가 비용이다

"자꾸 바꾸면 현장이 피곤하다"고 뭉뚱그려 생각하던 것에 이름이 있었다.  

찾아보니 nervousness는 원래 MRP(자재소요계획)에서 나온 말로, *"MRP 계획의 유의미한 변경"* 또는 *"불안정성"*으로 정의됐다고 한다.  
그게 스케줄링으로 넘어와 잦은 스케줄 변경 자체를 지칭하는 용어가 됐다.  

### stability 와 robustness 는 다르다

옆에 붙는 개념이 둘 더 있다.  

* stability — 스케줄이 실행되는 동안 겪는 수정과 변경의 횟수를 재는 지표다  
* robustness — 교란이 있어도 성능이 크게 떨어지지 않는 성질  

셋을 구분해서 보게 된 게 이 논문에서 얻은 첫 소득이다.  
나는 "안정적인 스케줄"이라는 말을 셋을 섞어서 쓰고 있었다.  
변경 횟수가 적은 것(stability)과 교란에 성능이 안 무너지는 것(robustness)은 다른 이야기이고, **둘은 상충할 수도 있다.**  

<br>
<br>
<br>

## 프레임워크 — 네 단계로 나눠 묻는다

논문의 뼈대는 재스케줄링을 네 층으로 쪼갠 것이다.  
각 층이 서로 다른 질문에 답한다.  

| 층 | 답하는 질문 | 원문 정의 |
|---|---|---|
| Environment | 어떤 작업들을 대상으로 하는가 | 스케줄에 포함되어야 할 job의 집합을 정한다 |
| Strategy | 애초에 스케줄을 만드는가 | 생산 스케줄을 생성하는지 여부를 기술한다 |
| Policy | 언제 다시 짜는가 | 재스케줄링이 언제 일어나야 하는지 규정한다 |
| Method | 어떻게 고치는가 | 스케줄을 어떻게 생성하고 갱신하는지 기술한다 |

이 분리가 유용한 이유는, 내가 헷갈렸던 용어들이 **서로 다른 층에 속해 있었기 때문**이다.  
dynamic scheduling은 Strategy 층이고, event-driven은 Policy 층이고, partial rescheduling은 Method 층이다.  
같은 평면에 놓고 비교하니 당연히 정리가 안 됐던 것이다.  

### Strategy — 스케줄을 만드느냐 마느냐

두 가지다.  

* dynamic scheduling — 스케줄을 아예 만들지 않는다. 그때그때 규칙으로 다음 작업을 고른다  
* predictive-reactive scheduling — 미리 스케줄을 만들어 두고, 교란이 생기면 수정한다  

논문은 제조 시스템에서는 predictive-reactive가 가장 흔히 쓰이는 접근이라고 정리한다.  

### Policy — 언제 다시 짜느냐

* periodic — 주기적으로 다시 짠다  
* event-driven — 설비 고장, 긴급 주문, 주문 취소 같은 사건이 생기면 다시 짠다  
* hybrid — 주기적으로 갱신하되, 사건이 생기면 그때 다시 짠다  

읽다가 걸린 게 하나 있다.  
원문은 *"세 가지 정책"*이라고 써놓고 periodic, event-driven, continuous, hybrid 네 개를 나열한다.  
바로 뒤 문장에서 continuous(=dynamic)를 event-driven의 특수한 경우라고 설명하니 실질은 셋이 맞는데, 문장만 보면 어긋난다.  
서베이 논문에서도 이런 게 남는구나 싶었다.  

<br>
<br>
<br>

### Method — 고치는 방법 세 가지

가장 실무적인 부분이다. 교란으로 스케줄이 실행 불가능해졌을 때 쓰는 방법이 셋이다.  

| 방법 | 무엇을 하나 | 변경량 |
|---|---|---|
| right-shift rescheduling | 남은 작업 전부를 필요한 시간만큼 뒤로 민다 | 가장 적다 |
| partial rescheduling | 교란에 직간접으로 영향받은 작업만 다시 짠다 | 중간 |
| regeneration | 남은 작업 전부를 처음부터 다시 짠다 | 가장 많다 |

right-shift의 원문 정의가 명쾌하다. 간트 차트에서 남은 각 작업을 스케줄이 실행 가능해지는 데 필요한 시간만큼 오른쪽으로 미는 것이다.  
설비가 2시간 멈췄으면 뒤 작업을 전부 2시간씩 미는 식이다. 순서는 하나도 안 바뀌니 현장 혼란이 최소다. 대신 최적성은 포기한다.  

partial rescheduling은 *affected operations rescheduling*이라고도 불린다고 한다.  
영향받은 것만 건드린다는 발상이 곧 minimal perturbation이고, 내가 하려던 게 이거였다.  

<br>
<br>
<br>

### 그래서 내 접근은 어느 칸인가

이 논문을 읽고 가장 뼈아팠던 건 여기다.  

내가 만들려던 구조는 평시에는 규칙으로 dispatch하다가 장애가 나면 optimizer를 부르는 것이다.  
그런데 이 프레임워크에 넣어보면 이렇게 된다.  

* Strategy — predictive-reactive  
* Policy — event-driven  
* Method — partial rescheduling  

**세 칸 다 이미 있는 칸이다.** 새로 만든 게 하나도 없다.  

이건 나쁜 소식이자 좋은 소식이다.  
나쁜 건 "장애 시 부분 재스케줄"이라는 구성만으로는 신규성을 주장할 수 없다는 것이다. 2003년 서베이에 이미 분류돼 있는 조합이다.  
좋은 건 **내 설계가 표준 분류의 어느 칸인지 한 문장으로 말할 수 있게 됐다는 것**이다. 그게 없으면 논문 심사에서 첫 질문에 막힌다.  

차별화를 하려면 칸 자체가 아니라 칸 안에서 찾아야 한다.  
partial rescheduling을 무엇으로 푸는가, 어디까지를 "영향받은 작업"으로 볼 것인가, 그 경계를 어떻게 증명하는가 — 이런 것들이다.  

<br>
<br>
<br>

## 지난 글과 이어지는 지점

읽으면서 계속 [lex 최소화 글](/optimization/lex-minimization-cpsat/)이 떠올랐다.  

거기서 나는 상위 목적을 등식으로 동결하고 그 안에서 하위 목적을 최적화했다.  
`model.Add(f1 == f1_star)` 한 줄로 "1순위는 손해 보지 않는다"를 보장한 것이다.  

minimal perturbation은 구조가 같다.  
**이전 스케줄을 기준점으로 두고 그로부터의 이탈을 최소화**한다. 동결하는 대상이 목적함수 값이냐 이전 해냐만 다르다.  

그러고 보니 nervousness도 lex 글에서 다룬 문제와 같은 얼굴이다.  
그 글에서 같은 입력에 결과가 매번 달라지는 걸 문제 삼았는데, 그건 입력이 안 바뀌었는데도 해가 흔들리는 경우였다.  
nervousness는 입력이 조금 바뀌었을 때 해가 크게 흔들리는 경우다. 둘 다 "해의 안정성"이라는 하나의 축 위에 있다.  

<br>
<br>
<br>

## 정리

* 자꾸 바꾸면 현장이 피곤하다고 뭉뚱그려 부르던 것에 nervousness라는 이름이 있었다. MRP에서 건너온 말이다  
* stability(변경 횟수)와 robustness(성능 유지)는 다른 지표이고 상충할 수 있다. 나는 둘을 섞어 쓰고 있었다  
* 재스케줄링은 Environment / Strategy / Policy / Method 네 층으로 나뉜다. 내가 헷갈린 용어들은 서로 다른 층에 있었다  
* 고치는 방법은 right-shift(전부 밀기) / partial(영향받은 것만) / regeneration(전부 다시)이고, 변경량이 이 순서로 커진다  
* **내 설계는 predictive-reactive + event-driven + partial 이었다. 세 칸 다 이미 있는 칸이다**  
* 새 조합을 만든 게 아니므로, 신규성은 칸 밖이 아니라 칸 안에서 찾아야 한다  

아직 못 본 것도 적어둔다.  
이 논문은 분류 체계를 주지만 어느 방법이 언제 유리한지는 답하지 않는다. right-shift와 partial 사이 어디를 골라야 하는지는 결국 재봐야 아는 문제로 보인다.  
그리고 "영향받은 작업"의 경계를 어떻게 정하느냐가 partial rescheduling의 전부인데, 그 정의를 이 논문은 각 연구에 맡긴다.  
다음은 시뮬레이터로 설비를 실제로 죽여서 right-shift와 partial의 변경량과 지연을 나란히 재보려 한다. 그래야 이 분류가 내 문제에서 무슨 의미인지 알 수 있다.  
