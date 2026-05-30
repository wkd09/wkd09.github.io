---
title: "Normalization과 PCA 정리"
date: 2026-05-29 13:50:00 +0900
categories:
  - study
tags:
  - DeepLearning
  - Normalization
  - PCA
source: "Notion PDF Export - Normalization, PCA"
---

![internal covariate shift](/assets/images/blog/internal-covariate-shift.png)
![batch normalization](/assets/images/blog/batch-normalization.png)
![normalization formulas](/assets/images/blog/normalization-formulas.png)
![batch norm algorithm](/assets/images/blog/batch-norm-algorithm.png)

![Normalization and PCA](/assets/images/blog/normalization-pca.svg)

<small>Image: [Wikimedia Commons - Gaussian scatter PCA](https://commons.wikimedia.org/wiki/Special:FilePath/GaussianScatterPCA.svg)</small>

Normalization은 학습을 안정적으로 만들기 위해 값의 분포를 조정하는 방법이다. PCA는 데이터의 중요한 분산 방향을 찾아 차원을 줄이는 방법이다. 둘 다 데이터를 더 다루기 쉬운 형태로 바꾼다는 공통점이 있다.

## Batch Normalization

Batch Normalization은 mini-batch 단위로 activation의 평균과 분산을 맞춘다.

$$
\hat{x}_i
= \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

이후 학습 가능한 $\gamma$, $\beta$를 적용한다.

$$
y_i = \gamma \hat{x}_i + \beta
$$

Batch Normalization의 목적은 layer를 통과할 때 activation 분포가 계속 바뀌는 문제를 줄이는 것이다. 이를 internal covariate shift 관점에서 설명한다. 분포가 안정되면 더 큰 learning rate를 사용할 수 있고, 학습이 빨라지며, 어느 정도 regularization 효과도 생긴다.

훈련 중에는 mini-batch의 평균과 분산을 사용한다. 추론 중에는 학습 과정에서 누적한 running mean과 running variance를 사용한다. 이 차이를 이해하지 못하면 train과 inference 결과가 달라지는 문제를 디버깅하기 어렵다.

Batch Normalization은 batch size가 충분히 클 때 안정적으로 동작한다. batch size가 너무 작으면 batch 통계가 noisy해져 오히려 성능이 흔들릴 수 있다. 이 경우 Group Normalization이나 Layer Normalization을 고려할 수 있다.

또한 fine-tuning에서는 BatchNorm layer를 어떻게 다룰지도 중요하다. pretrained CNN을 작은 데이터셋에 맞출 때 running statistics를 계속 업데이트하면 성능이 나빠질 수 있어, BatchNorm을 freeze하는 전략을 쓰기도 한다.

## Layer Normalization

Layer Normalization은 batch 방향이 아니라 하나의 sample 내부 hidden dimension을 기준으로 정규화한다.

Batch Normalization은 batch size가 작거나 sequence 길이가 바뀌는 문제에서 불안정할 수 있다. 반면 Layer Normalization은 각 token 또는 sample 내부에서 계산되므로 batch size에 덜 민감하다.

Transformer 계열 모델에서 Layer Normalization이 많이 쓰이는 이유도 여기에 있다. sequence 모델은 batch와 time dimension의 구조가 복잡하기 때문에, layer 단위로 안정화하는 방식이 더 잘 맞는다.

Transformer에서는 LayerNorm의 위치도 중요하다. Post-LN 구조는 attention이나 feed-forward block 뒤에 normalization을 두고, Pre-LN 구조는 block 앞에 normalization을 둔다. 깊은 Transformer에서는 Pre-LN이 gradient 흐름을 더 안정적으로 만들어 학습이 쉬운 경우가 많다.

## Batch Norm과 Layer Norm 비교

Batch Normalization은 같은 feature를 batch 전체에 대해 정규화한다. CNN처럼 batch 통계가 안정적인 구조에서 자주 사용된다.

Layer Normalization은 한 sample의 hidden vector 전체를 정규화한다. RNN, Transformer처럼 sequence를 다루는 모델에서 자주 사용된다.

정리하면 Batch Norm은 batch 통계에 의존하고, Layer Norm은 sample 내부 통계에 의존한다.

간단히 기억하면 CNN에서는 Batch Norm을 먼저 떠올리고, Transformer나 sequence model에서는 Layer Norm을 먼저 떠올리면 된다. 물론 모델 구조와 batch size에 따라 예외는 있다.

## PCA

PCA는 Principal Component Analysis의 약자다. 고차원 데이터를 더 낮은 차원으로 투영하면서도 데이터의 중요한 정보를 최대한 보존하려는 방법이다.

핵심은 데이터의 분산이 가장 큰 방향을 찾는 것이다. 첫 번째 principal component는 데이터가 가장 많이 퍼져 있는 방향이고, 두 번째 principal component는 첫 번째와 직교하면서 다음으로 분산이 큰 방향이다.

PCA는 다음과 같은 상황에서 유용하다.

- feature 수가 너무 많아 시각화가 어려울 때
- 노이즈가 많은 차원을 줄이고 싶을 때
- 모델 학습 전에 차원을 줄여 계산 비용을 낮추고 싶을 때

단점도 있다. PCA는 선형 투영이기 때문에 복잡한 비선형 구조를 잘 보존하지 못할 수 있다. 또한 principal component는 원래 feature의 조합이므로 해석이 어려워질 수 있다.

PCA를 적용하기 전에는 보통 feature scaling을 먼저 한다. PCA는 분산이 큰 방향을 찾기 때문에, 단위가 큰 feature가 principal component를 지배할 수 있다. 예를 들어 키와 연봉처럼 scale이 전혀 다른 feature를 그대로 넣으면 분산이 큰 feature가 과하게 반영된다.

PCA 결과를 해석할 때는 explained variance ratio를 본다. 첫 번째, 두 번째 principal component가 전체 분산의 몇 퍼센트를 설명하는지 확인하면 차원을 얼마나 줄여도 되는지 판단할 수 있다. 예를 들어 2개의 component가 90% 이상의 분산을 설명한다면 시각화나 간단한 모델링에 충분할 수 있다.

## 언제 어떤 방법을 쓸까

Normalization은 모델 학습 중 activation 분포를 안정화하는 데 쓰인다. 즉, 학습이 흔들리거나 깊은 network를 안정적으로 훈련하고 싶을 때 중요하다.

PCA는 학습 전 데이터 분석이나 feature 전처리에 가깝다. feature 수가 많아 구조를 보기 어렵거나, 차원을 줄여 계산량을 낮추고 싶을 때 사용한다.

둘 다 데이터 분포를 다루지만 위치가 다르다. Normalization은 network 내부 학습 안정화에 가깝고, PCA는 입력 feature 공간을 재구성하는 방법에 가깝다.

Normalization과 PCA 모두 데이터를 바꾸는 과정이지만 목적이 다르다. Normalization은 학습 안정성을 높이는 것이고, PCA는 차원을 줄이면서 중요한 분산 구조를 보존하는 것이다.
