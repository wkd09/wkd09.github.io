---
title: "Loss Function 정리: 모델이 얼마나 틀렸는지 측정하는 방법"
date: 2026-05-29 13:20:00 +0900
last_modified_at: 2026-07-04 00:00:00 +0900
categories:
  - study
tags:
  - ML
  - Loss
  - Optimization
source: "Notion PDF Export - Loss Function"
---

<small>Image: [Wikimedia Commons - Simplified neural network training example](https://commons.wikimedia.org/wiki/Special:FilePath/Simplified_neural_network_training_example.svg)</small>

# Loss Function

## 손실 함수란?

손실 함수는 딥러닝에서 모델이 얼마나 틀렸는지 나타내는 지표이다. 모델의 예측값과 실제값의 차이를 측정하며, 손실 함수가 작을수록 모델의 예측이 실제값에 더 가깝다. 반대로 손실 함수가 크면 실제값과 멀다는 뜻이다.

---

## 손실 함수의 역할은?

1. 모델의 최적화: 딥러닝 모델을 학습시킬 때, 모델의 파라미터와 편향을 조정해 손실 함수를 최소화하는 것을 목표로 한다. 이 과정에서 경사 하강법을 사용한다.
2. 모델 평가: 모델이 훈련 데이터에 대해 잘 학습하고 있는지 판단하기 위해 손실 함수를 모니터링하고, 과적합이나 과소적합 문제를 감지한다.

---

## 손실함수의 종류

회귀 문제:

- MSE(Mean Squared Error, 평균 제곱 오차): 실제값과 예측값의 차이를 제곱한 뒤 평균을 구하는 함수이다. 회귀 문제에서 많이 사용된다.

분류 문제:

- Cross-Entropy Loss(교차 엔트로피 손실): 다중 클래스 분류에서 사용되며, 예측 확률 분포와 실제 분포 사이의 차이를 측정한다.
- Binary Cross-Entropy: 이진 분류에서 사용되며, 실제값과 예측값이 0 또는 1인 경우의 손실을 계산한다.

---

### Cross-Entropy Loss

Cross-Entropy Loss는 주로 분류에 사용되는 손실 함수로, 모델의 예측값과 실제값 사이의 차이를 측정한다. 예측이 실제와 얼마나 일치하는지를 확률적인 관점에서 평가하며, 모델 출력 분포가 실제 분포와 얼마나 가까운지 본다.

![cross entropy loss](/assets/images/blog/cross-entropy-loss.png)

### 1. Binary Cross-Entropy Loss

이 함수는 이진 분류 문제에서 사용된다. 두 클래스로 나누어진 데이터를 분류하는 데 적합하다. 이진 분류 모델에서는 출력 노드가 하나이며, 이 노드의 출력은 클래스 1일 확률로 해석한다.

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left(y_i\log p_i + (1-y_i)\log(1-p_i)\right)
$$

![binary cross entropy math](/assets/images/blog/binary-cross-entropy-math.png)

- $N$: 샘플 수(batch size)
- $y_i$: 샘플 $i$의 실제 라벨(0 또는 1)
- $p_i$: 샘플 $i$에 대해 모델이 예측한 클래스 1의 확률

이 수식은 예측 확률 $p_i$가 실제 라벨 $y_i$와 얼마나 일치하는지 측정한다.

### 2. Categorical Cross-Entropy Loss

이 함수는 다중 클래스 문제에서 사용된다. 이 경우 출력 노드는 클래스 수와 동일하며, 각 클래스에 속할 확률을 출력한다.

$$
\mathcal{L}_{\text{CCE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} y_{ij}\log p_{ij}
$$

![categorical cross entropy math](/assets/images/blog/categorical-cross-entropy-math.png)

- $N$: 샘플 수(batch size)
- $C$: 클래스 수
- $y_{ij}$: 샘플 $i$의 실제 클래스 라벨(원-핫 인코딩으로 표현되며, 해당 클래스에만 1이고 나머지는 0)
- $p_{ij}$: 샘플 $i$에 대해 모델이 예측한 클래스 $j$의 확률

이 수식은 모델이 예측한 확률 분포가 실제 라벨(원-핫 인코딩)의 분포와 얼마나 일치하는지 측정한다.

### **3. Sparse Categorical Cross-Entropy Loss**

이 함수는 위 함수와 유사하지만, 라벨을 원-핫 인코딩 없이 정수로 제공할 때 사용한다. 원-핫 벡터를 만들지 않기 때문에 메모리를 절약할 수 있다.

$$
\mathcal{L}_{\text{SCCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}
$$

![sparse categorical cross entropy math](/assets/images/blog/sparse-cross-entropy-math.png)

- $y_i$: 샘플 $i$의 실제 클래스 라벨(정수로 표현됨)
- $p_{i,y_i}$: 샘플 $i$에 대해 모델이 예측한 실제 클래스 $y_i$의 확률

이 수식은 Categorical Cross-Entropy와 동일한 방식으로 동작하지만, 라벨이 원-핫 인코딩되지 않고 정수로 주어진다는 차이점이 있다.

---

## **MSE(Mean Squared Error)**

주로 회귀 문제에서 사용되는 손실 함수로, 모델의 예측값과 실제값 사이의 차이를 제곱한 후 평균을 구한 값이다. MSE는 예측값이 실제값에서 얼마나 떨어져 있는지 직관적으로 표현한다.

![mse loss](/assets/images/blog/mse-loss.png)

### 1. MSE**(Mean Squared Error)**

MSE는 가장 기본적인 손실 함수로, 예측값과 실제값 사이의 차이를 제곱하고 평균을 구하는 방식이다.

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

![mse math](/assets/images/blog/mse-math.png)

- $N$: 샘플의 수(batch size)
- $y_i$: 실제값(타겟)
- $\hat{y}_i$: 모델이 예측한 값

MSE는 각 데이터 포인트에 대해 예측값과 실제값의 차이를 계산하고 이를 제곱하여 평균을 내는 방식이다. 제곱을 사용하면 오차가 모두 양수가 되고, 특히 큰 오차에 더 큰 패널티를 부여한다. 예측값이 실제값과 가까워지면 손실 함수는 작아진다.

### 2. RMSE(Root Mean Squared Error)

RMSE는 MSE에서 제곱근을 취한 형태로, 오차를 실제값과 동일한 단위로 표현하는 데 유용하다.

$$
\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
$$

![rmse math](/assets/images/blog/rmse-math.png)

- $N$: 샘플의 수(batch size)
- $y_i$: 실제값(타겟)
- $\hat{y}_i$: 모델이 예측한 값

RMSE는 MSE에 비해 더 직관적인 표현이 가능하다. 예측 오차를 원래 데이터와 동일한 단위로 표현할 수 있기 때문에, 예측값이 실제값에 얼마나 가까운지 실질적인 단위로 보여준다.

### 3. MAE(Mean Absolute Error)

MAE는 예측값과 실제값 사이의 차이를 절댓값으로 변환한 후 평균을 구하는 방식이다.

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
$$

![mae math](/assets/images/blog/mae-math.png)

- $|y_i - \hat{y}_i|$: 샘플의 예측값과 실제값 사이의 절대 오차를 의미한다.

MSE와의 차이점은 제곱을 하지 않기 때문에 큰 오차에 덜 민감하다는 점이다. 따라서 MAE는 이상치(outlier)에 덜 영향을 받는다는 장점이 있다.
