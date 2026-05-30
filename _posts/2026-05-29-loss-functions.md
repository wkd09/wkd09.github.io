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

![mse loss](/assets/images/blog/mse-loss.png)
![cross entropy loss](/assets/images/blog/cross-entropy-loss.png)

![Loss and optimization loop](/assets/images/blog/loss-optimization.svg)

<small>Image: [Wikimedia Commons - Simplified neural network training example](https://commons.wikimedia.org/wiki/Special:FilePath/Simplified_neural_network_training_example.svg)</small>

Loss function은 모델의 예측값과 실제 정답 사이의 차이를 숫자로 표현하는 함수다. 학습은 이 loss를 줄이는 방향으로 파라미터를 업데이트하는 과정이라고 볼 수 있다.

모델이 아무리 복잡해도 학습의 기본 흐름은 단순하다. 예측을 만들고, 정답과 비교해 loss를 계산하고, 그 loss를 줄이는 방향으로 gradient를 구한 뒤 파라미터를 업데이트한다.

## 회귀에서 쓰는 손실 함수

회귀 문제는 연속적인 값을 예측한다. 따라서 예측값과 실제값의 거리 자체가 중요하다.

### MSE

Mean Squared Error는 오차를 제곱해서 평균낸다.

$$
\mathrm{MSE}
= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

오차를 제곱하므로 큰 오차에 더 큰 penalty를 준다. 이상치에 민감하지만, 미분이 깔끔해서 많이 사용된다.

### RMSE

Root Mean Squared Error는 MSE에 루트를 씌운 값이다.

$$
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}
$$

단위가 원래 target과 같아져 해석이 쉽다. 예를 들어 집값 예측에서 RMSE가 1000만 원이면 평균적으로 그 정도 규모의 오차가 난다고 이해할 수 있다.

### MAE

Mean Absolute Error는 오차의 절댓값을 평균낸다.

$$
\mathrm{MAE}
= \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

MSE보다 이상치에 덜 민감하다. 큰 오차를 특별히 더 강하게 벌주지 않기 때문에, outlier가 많은 데이터에서는 MAE가 더 안정적일 수 있다.

MSE와 MAE 중 무엇을 쓸지는 문제의 비용 구조와 연결된다. 큰 오차 하나가 서비스 품질에 치명적이면 MSE가 적절할 수 있고, 전체적으로 안정적인 중앙값 예측이 중요하면 MAE가 더 자연스럽다.

## 분류에서 쓰는 손실 함수

분류 문제에서는 모델이 정답 클래스에 얼마나 높은 확률을 줬는지가 중요하다. 이때 많이 쓰는 것이 cross entropy다.

### Cross Entropy

Cross entropy는 정답 분포와 예측 분포의 차이를 측정한다. 정답 클래스의 확률을 낮게 예측할수록 loss가 커진다.

이진 분류에서는 Binary Cross Entropy를 쓴다.

$$
\mathrm{BCE}
= -\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
$$

다중 분류에서는 Categorical Cross Entropy를 사용한다. 정답이 one-hot encoding이면 categorical cross entropy를 쓰고, 정답이 class index 형태이면 sparse categorical cross entropy를 쓴다. 수학적으로 목표는 비슷하지만 입력 label 형식이 다르다.

Cross entropy를 사용할 때 중요한 점은 모델 출력이 확률 분포처럼 해석되어야 한다는 것이다. 이진 분류에서는 sigmoid를 거친 확률을 쓰고, 다중 분류에서는 softmax를 통해 전체 class 확률 합이 1이 되도록 만든다.

실제 구현에서는 numerical stability 때문에 `softmax`와 `log`를 따로 계산하지 않고, framework가 제공하는 `CrossEntropyLoss`처럼 log-softmax가 내부적으로 결합된 함수를 쓰는 편이 안전하다.

## Class Imbalance가 있을 때

분류 데이터에서 class 비율이 심하게 불균형하면 일반 cross entropy만으로는 소수 class를 잘 학습하지 못할 수 있다. 모델이 다수 class만 맞춰도 loss를 꽤 낮출 수 있기 때문이다.

이럴 때는 class weight를 주거나 focal loss 같은 변형을 사용할 수 있다. class weight는 소수 class를 틀렸을 때 더 큰 penalty를 주는 방식이고, focal loss는 쉬운 sample의 영향은 줄이고 어려운 sample에 더 집중하도록 만든다.

## Loss와 Metric은 다르다

Loss는 학습 중 optimizer가 줄이는 값이고, metric은 사람이 모델을 평가하기 위해 보는 지표다. 둘은 같을 수도 있지만 항상 같지는 않다.

예를 들어 분류 모델은 cross entropy로 학습하지만, 실제 평가는 accuracy, F1-score, ROC-AUC로 볼 수 있다. 추천 시스템은 pairwise loss로 학습하면서 NDCG나 recall@k를 metric으로 볼 수 있다.

따라서 모델이 학습 중 loss는 내려가는데 validation metric이 좋아지지 않는다면, loss가 실제 목표를 잘 대변하고 있는지 확인해야 한다.

## 좋은 loss를 고르는 기준

Loss function은 단순히 수식 문제가 아니다. 어떤 오차를 더 심각하게 볼 것인지 정하는 기준이다.

예를 들어 이상치가 큰 문제에서는 MSE가 outlier에 지나치게 끌릴 수 있다. 반면 큰 오차를 반드시 강하게 줄여야 하는 문제라면 MSE가 적절할 수 있다. 분류에서는 확률 calibration과 class imbalance까지 같이 고려해야 한다.

좋은 loss를 고르는 기준은 다음처럼 정리할 수 있다.

- 예측 대상이 연속값인지 class인지 확인한다.
- 큰 오차를 더 강하게 벌줄지, 모든 오차를 비슷하게 볼지 정한다.
- class imbalance나 outlier처럼 데이터 분포의 특성을 반영한다.
- 학습 loss와 실제 서비스 metric이 어긋나지 않는지 확인한다.

정리하면 loss function은 모델의 목표를 숫자로 정의하는 장치다. 좋은 모델을 만들려면 모델 구조뿐 아니라 어떤 loss를 최소화하고 있는지까지 이해해야 한다.
