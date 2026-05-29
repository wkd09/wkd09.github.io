---
title: "Backpropagation 정리: Chain Rule로 가중치를 업데이트하는 방법"
date: 2026-05-29 13:40:00 +0900
categories:
  - study
tags:
  - DeepLearning
  - Backpropagation
  - Gradient
source: "Notion PDF Export - Back propagation"
---

![Backpropagation through a neural network](/assets/images/blog/backpropagation.svg)

<small>Image: [Wikimedia Commons - Artificial neural network](https://commons.wikimedia.org/wiki/Special:FilePath/Artificial_neural_network.svg)</small>

Backpropagation은 neural network가 loss를 줄이기 위해 각 weight를 얼마나 바꿔야 하는지 계산하는 방법이다. 핵심은 **chain rule**이다.

Forward propagation에서는 입력이 layer를 지나 output을 만든다. 이후 loss function이 예측값과 정답의 차이를 계산한다. Backpropagation은 이 loss가 각 weight에 의해 얼마나 변하는지 뒤에서 앞으로 거슬러 올라가며 계산한다.

## Forward Propagation

간단한 2-layer network를 생각하면 입력 `x`는 hidden layer를 거쳐 output으로 변환된다.

$$
z_1 = W_1x + b_1
$$

$$
a_1 = f(z_1)
$$

$$
z_2 = W_2a_1 + b_2
$$

$$
\hat{y} = g(z_2)
$$

여기서 `f`, `g`는 activation function이다. 예측값 `y_hat`이 나오면 MSE나 cross entropy 같은 loss를 계산한다.

Forward pass에서 중요한 것은 각 연산의 중간값을 저장해둔다는 점이다. Backward pass는 이 중간값을 사용해 gradient를 계산한다. 예를 들어 ReLU를 통과한 값이 0 이하였는지, sigmoid의 출력이 얼마였는지 알아야 local gradient를 구할 수 있다.

## Chain Rule

어떤 weight `w`가 loss에 미치는 영향을 직접 계산하기 어려울 때, 중간 변수들을 곱해서 gradient를 구할 수 있다.

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial z}
\cdot
\frac{\partial z}{\partial w}
$$

이것이 chain rule이다. 딥러닝 모델은 layer가 많기 때문에 각 layer의 local gradient를 저장해두고 뒤에서부터 곱하면서 gradient를 전달한다.

계산 그래프로 보면 더 직관적이다. 하나의 node는 연산이고 edge는 tensor의 흐름이다. Backpropagation은 output loss에서 시작해 그래프를 거꾸로 따라가며 각 node의 gradient를 계산한다.

예를 들어 `z = x * y`라면 `z`에 대한 `x`의 gradient는 `y`이고, `y`의 gradient는 `x`다. 이런 작은 local derivative를 전체 그래프에 연결하면 큰 neural network의 gradient도 계산할 수 있다.

## Weight Update

Gradient를 구한 뒤 optimizer가 weight를 업데이트한다.

$$
w_{t+1}
= w_t - \eta \frac{\partial L}{\partial w_t}
$$

gradient가 양수라면 weight를 줄이고, 음수라면 weight를 늘린다. 이렇게 하면 loss가 감소하는 방향으로 이동한다.

여기서 $\eta$가 learning rate다. gradient 방향이 맞더라도 learning rate가 너무 크면 최소점을 지나쳐 발산할 수 있고, 너무 작으면 학습이 지나치게 느려진다. 그래서 backpropagation은 optimizer, learning rate schedule과 함께 봐야 한다.

## Vanishing Gradient와 Exploding Gradient

Backpropagation은 gradient를 여러 layer에 걸쳐 곱하면서 전달한다. 이때 곱해지는 값들이 1보다 계속 작으면 gradient가 거의 0에 가까워진다. 이것이 vanishing gradient다.

반대로 곱해지는 값들이 계속 크면 gradient가 폭발적으로 커질 수 있다. 이것이 exploding gradient다. 이런 문제는 깊은 network나 긴 sequence를 다루는 RNN에서 특히 잘 나타난다.

이를 완화하기 위해 ReLU 계열 activation, residual connection, normalization, gradient clipping 같은 기법이 사용된다. Transformer에서 residual connection과 Layer Normalization이 중요한 이유도 gradient 흐름을 안정화하는 것과 관련이 있다.

## Autograd가 하는 일

PyTorch나 TensorFlow의 autograd는 forward pass 중 연산 그래프를 기록한다. 이후 `backward()`를 호출하면 loss에서 시작해 필요한 gradient를 자동으로 계산한다.

사용자는 직접 미분식을 모두 작성하지 않아도 되지만, `requires_grad`, `detach`, `no_grad` 같은 옵션을 잘못 사용하면 gradient가 끊길 수 있다. 학습이 되지 않을 때는 loss가 줄지 않는지만 볼 것이 아니라, 실제 parameter에 gradient가 들어오는지도 확인해야 한다.

## 왜 역전파가 중요한가

Neural network는 수많은 weight를 가진다. 각 weight가 loss에 미치는 영향을 일일이 수동으로 계산할 수 없다. Backpropagation은 계산 그래프를 따라 gradient를 효율적으로 재사용하므로, 큰 모델도 학습할 수 있게 해준다.

현대 딥러닝 프레임워크의 autograd는 이 과정을 자동으로 수행한다. 하지만 내부 원리를 이해하면 exploding gradient, vanishing gradient, learning rate 문제를 디버깅할 때 훨씬 명확하게 볼 수 있다.
