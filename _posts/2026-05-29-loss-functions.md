---
title: "Loss Function 정리: 모델이 얼마나 틀렸는지 측정하는 방법"
date: 2026-05-29 13:20:00 +0900
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

손실함수는 딥러닝에서 모델 성능을 나타내는 지표이다. 이는 모델의 예측 값과 실제 값의 차이를 측정한다. 손실 함수가 작을수록 모델의 예측이 실제 값에 더 가까워지고(성능 좋음), 반대로 손실 함수가 크면 실제 값과 멀어진다(성능 안좋음).

---

## 손실 함수의 역할은?

1. 모델의 최적화: 딥러닝 모델을 학습 시킬 때, 모델의 파라미터와 편향을 조정해 손실함수 최소화 하는것을 목표로 한다. 이 과정에서 경사 하강법 사용
2. 모델 평가: 모델이 훈련 데이터에 대해 잘 학습하고 있는 판단하기 위해, 손실함수 모니터링을 통해 과적합, 과소적합 문제를 감지

---

## 손실함수의 종류

- 회귀 문제
1. MSE(Mean Squared Error, 평균 제곱 오차) : 실제 값과 예측 값의 차이를 제곱한 뒤 평균을 구하는 함수이다. 회귀 문제에서 많이 사용된다.
- 분류 문제
1. Cross-Entropy Loss(교차 엔트로피 손실) : 다중 클래스 분류에서 사용됨, 예측 확률 분포와 실제 분포 사이의 차이를 측정한다.
2. Binary Cross-Entropy : 이진 분류에서 사용되며, 실제 값과 예측 값이 0 또는 1인 경우의 손실을 계산

---

### Cross-Entropy Loss

Cross-Entropy Loss는 주로 분류에 사용되는 손실함수로, 모델의 예측 값과 실제값 사이 차이를 측정하는데 사용, 예측이 실제와 얼마나 일치하는지를 확률적인 관점에서 평가함. 모델 출력 분포가 실제 분포와 얼마나 가까운지 평가 한다.

![cross entropy loss](/assets/images/blog/cross-entropy-loss.png)

### 1. Binary Cross-Entropy Loss

이 함수는 이진 분류 문제에서 사용된다. 두 클래스로 나누어진 데이터를 분류하는 데 적합하다. 이진 분류 모델에서는 출력 노드가 하나이며 이 노드의 출력은 클래스 1일 확률로 해석.

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left(y_i\log p_i + (1-y_i)\log(1-p_i)\right)
$$

![binary cross entropy math](/assets/images/blog/binary-cross-entropy-math.png)

- N : 샘플 수 (batch size)
- y_i : 샘플 i의 실제 라벨(0 또는 1)
- p_i : 샘플 i의 모델이 예측한 클래스 1의 확률

이 수식은 예측 확률 p_i 가 실제 라벨 y_i 와 얼마나 일치하는지 측정

### 2. Categorical Cross-Entropy Loss

이 함수는 다중 클래스 문제에서 사용된다. 이 경우, 출력 노드는 클래스의 수와 동일하며, 각 클래스에 속할 확률을 출력함.

$$
\mathcal{L}_{\text{CCE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} y_{ij}\log p_{ij}
$$

![categorical cross entropy math](/assets/images/blog/categorical-cross-entropy-math.png)

- N : 샘플 수 (batch size)
- C : 클래스 수
- y_ij : 샘플 i의 실제 클래스 라벨(원-핫 인코딩으로 표현됨, 즉 해당 클래스에만 1, 나머지는 0)
- p_ij : 샘플 i에 대해 모델이 예측한 클래스 j의 확률

이 수식은 모델이 예측한 확률 분포가 실제 라벨(원-핫 인코딩)의 분포와 얼마나 일치하는 지 측정

### **3. Sparse Categorical Cross-Entropy Loss**

이 함수는 위 함수와 유사하지만, 라벨을 원-핫 인코딩 없이 정수로 제공 할때 사용. 메모리 절약 가능.

$$
\mathcal{L}_{\text{SCCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}
$$

![sparse categorical cross entropy math](/assets/images/blog/sparse-cross-entropy-math.png)

- y_i : 샘플 i의 실제 클래스 라벨 (정수로 표현됨)
- p_i,yi : 샘플 i에 대해 모델이 예측한 실제 클래스 y_i 의 확률

이 수식은 Categorical Cross-Entropy 와 동일한 방식으로 동작하지만, 라벨이 원-핫 인코딩 되지 않고 정수로 주어진다는 차이점이 있다.

---

## **MSE(Mean Squared Error)**

주로 회귀 문제에서 사용되는 손실함수로, 모델의 예측 값과 실제 값 사이의 차이를 제곱한 후 평균 값을 구한 값이다. MSE는 예측 값이 실제 값에 얼마나 떨어져 있는지 직관적으로 표현한다.

![mse loss](/assets/images/blog/mse-loss.png)

### 1. MSE**(Mean Squared Error)**

MSE는 가장 기본적인 손실함수로, 예측 값과 실제 값 사이의 차이를 제곱하고 평균을 구하는 방식이다.

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

![mse math](/assets/images/blog/mse-math.png)

- N : 샘플의 수 (batch size)
- y_i : 실제 값(타겟)
- hat{y_i} : 모델이 예측한 값

MSE는 각 데이터 포인트에 대해 예측 값과 실제 값의 차이를 계산하고 이를 제곱하여 평균을 내는 방식이다. 제곱하는 이유는 모두 양수로 만들어 오차를 극대화 시켜, 특히 큰 오차에 더 큰 패널티를 부여한다. 예측 값이 실제 값과 가까워지면 손실함수는 작아진다.

### 2. RMSE(Root Mean Squared Error)

RMSE는 MSE에서 제곱근을 취한 형태로, 오차를 실제 값과 동일한 차원으로 표현하는데 유용하다.

$$
\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
$$

![rmse math](/assets/images/blog/rmse-math.png)

- N : 샘플의 수 (batch size)
- y_i : 실제 값(타겟)
- hat{y_i} : 모델이 예측한 값

RMSE는 MSE에 비해 직관적인 표현이 가능하다. 예측 오차를 원래 데이터와 동일한 단위로 표현할수있다. RMSE는 예측값이 실제 값에 얼마나 가까운지 실질적인 단위로 보여준다.

### 3. MAE(Mean Absolute Error)

MAE는 예측값과 실제값 사이의 차이를 절대값으로 변환 후 평균을 구하는 방식이다.

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
$$

![mae math](/assets/images/blog/mae-math.png)

- | y_i - hat{y_i} | : 샘플의 예측 값과 실제 값 사이의 절대 오차를 의미한다.

MSE 와의 차이점은 제곱을 하지 않기 때문에 큰 오차에 대해 덜 민감하다. 따라서 MAE는 이상치(outlier)에 덜 영향을 받는다는 장점이 있다.
