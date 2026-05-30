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


# Back propagation

역전파란 target 값과 모델이 계산한 output이 얼마나 차이 나는지(loss 값) 구한 후 오차값을 다시 출력층에서 입력층으로 전파해서 각 노드가 가지고 있는 변수를 갱신하는 알고리즘이다.

그럼 각 노드가 가지고 있는 변수인 weight, bais 값을 어떻게 갱신하는지, 각 노드나 레이어가 가지고 있는 변수들의 값이 제각각인데 그 값들을 얼마나 변경하는지?

이러한 질문들을 아래 chain rule을 통해 해결한다.

## Chain Rule

Chain Rule은 미분에서 연쇄법칙이라고도 불린다. 먼저 수학적으로 정의를 해보자

함수 $f$, $g$가 있을 때 $f$와 $g$가 모두 미분이 가능하고 $F=f(g(x))=f∘g$로 정의된 합성 함수이면 $F$는 미분 가능하다.

이때 $F′(x)=f′(g(x))⋅g′(x)$이다.

$t=g(x)$라고 한다면, $\frac{dy}{dx} = \frac{dy}{dt} \frac{dt}{dx}$이 성립한다.

---

이런거만 보면 이해가 안된다..

먼저 합성함수는 그냥 어떤 함수의 인자로 다른 함수가 주어진 함수이다. 아래 코드를 보면 이해가 될 것이다.

```python
def f(g) {
  return g * 3
}
def g(x) {
  return x + 1
}

x = 3;
F = f(g(x))
// F = 12
```

그럼 미분이 가능하다는 말이 뭘까? 만약이 미분이  $X$와 $X'$간의 기울기를 구하는 정도로 이해했다면

합성함수에서 갑자기 왜 기울기를 구하지라는 생각이 들 것이다.

하지만 기울기를 구한다는 말은 변화량을 구한다는 것과 동일하다. 합성 함수 코드에서 $F$를 선언할 때 $g$에 주는 값즉 매개변수를 변경한다면 최종적으로 $F$값이 변한다는 것을 알 수 있다.

```python
F = f(g(4))
// F = 15
F = f(g(2))
// F = 9
```

즉 chain rule이란 $X$가 변화했을 때 함수 $g$가 얼마나 변하는지, 그리고 그로 인해 함수 $g$의 변화로 함수 $f$가 얼마나 변하는지 알 수 있고, 이러한 연쇄적인 변화를 알 수 있게 되었다면, 값 $F$의 변화량에 기여하는 함수 $f$, $$ $g$의 기여도를 알 수 있게 된다.

그렇다면 함수 여러개가 들어가면 어떻게 될까?

---

이변수함수  $z=f(x,y)$에서 $x=h(s,t),y=g(s,t)$ 일때 

$f(x,y),g(s,t),h(s,t)$ 가 모두 미분 가능하면

$$
\frac{\partial s}{\partial z} = \frac{\partial s}{\partial x} \frac{\partial x}{\partial z} + \frac{\partial s}{\partial y} \frac{\partial y}{\partial z} \quad \frac{\partial t}{\partial z} = \frac{\partial t}{\partial x} \frac{\partial x}{\partial z} + \frac{\partial t}{\partial y} \frac{\partial y}{\partial z}
$$

나타낼 수 있다.

$s$나 $t$가 얼만큼인지 모르지만 어쨋든 변했을 때, 함수 z의 변화량을 위 식 처럼 구할 수 있다.

그리고 $∂$는 편미분을 뜻하는 기호인데, 메인이 되는 변수 하나는 남겨두고, 나머지 변수는 그냥 개무시하는 미분법으로 $s$, $z$ 관계를 구하는 식에서 아예 $t$가 없는 것을 알 수 있다.

---

## Foward-propagation

Back-propagation이 이뤄지기 전 Foward-propagation을 진행해야 한다. 초기화 한 $w$값과 input인 $x$를 가지고 순전파 계산을 진행한 후 우리가 원하는 값이 나오는지, 나오지 않았다면 얼마나 차이 나는지 구하야 한다.

![backprop forward](/assets/images/blog/backprop-forward.png)

이 모델을 2개의 input, 2개의 output을 가지고 2개의 hidden-layer를 가진 2-Layer-NN 모델이다.

![backprop backward](/assets/images/blog/backprop-backward.png)

먼저 output으로 $y1$에 0.2, $y2$에 0.7으로 각각 넣어주고, 그리고 input으로는 $x1$에 0.2,  $x2$에 0.5를 넣어 주고, 각 $w$는 그냥 아무 값이나 넣었다.

난 이 계산에서 activation function으론 sigmoid, loss function으로 MSE를 사용한다.

먼저 layer0에서 받을 값부터 계산해보자, 보통 행렬로 계산한다.

$$
\begin{aligned}z_{10} &= \begin{bmatrix} x_1 & x_2 \end{bmatrix} \times \begin{bmatrix} w_{100} \\ w_{200} \end{bmatrix} \\z_{11} &= \begin{bmatrix} x_1 & x_2 \end{bmatrix} \times \begin{bmatrix} w_{110} \\ w_{210} \end{bmatrix}\end{aligned}
$$

저 행렬 곱을 풀어보면 아래 식과 같고 결국 $wx$들의 합의 형태이다.

$$
\begin{aligned}z_{10} &= x_1 w_{100} + x_2 w_{200} \\&= (0.2 \times 0.1) + (0.5 \times 0.3) \\&= 0.02 + 0.15 = 0.17 \\\\z_{11} &= x_1 w_{110} + x_2 w_{210} \\&= (0.2 \times 0.2) + (0.5 \times 0.1) \\&= 0.04 + 0.05 = 0.09\end{aligned}
$$

$z_{10}$와 $z_{11}$의 값을 구했으면 activation function에 넣어 $a_{10}$과 $a_{11}$ 값을 구해보자

사용할 sigmoid의 수식과 코드는 밑에 보이는 것 과 같다.

$$
\sigma = \frac{1}{1 + e^{-x}}
$$

```python
def sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
```

$$
\begin{aligned}a_{10} &= \sigma(z_{10}) = 0.54 \\a_{11} &= \sigma(z_{11}) = 0.52\end{aligned}
$$

다음 레이어도 위와 같이 구해보자

$$
\begin{aligned}&\text{[Layer 0]} && \text{[Layer 1]} \\z_{10} &= 0.17 \quad \to \quad a_{10} = 0.54 \quad & z_{20} &= 0.27 \quad \to \quad a_{20} = 0.57 \\z_{11} &= 0.09 \quad \to \quad a_{11} = 0.52 \quad & z_{21} &= 0.43 \quad \to \quad a_{21} = 0.61\end{aligned}
$$

$y_{1}$과 $y_{2}$는 각 $a_{20}$과 $a_{21}$으로 output을 구했다. 근데 우리가 원하는 $y_{1}$과 $y_{2}$는 0.2와 0.7이였는데, 우리가 구한 output은 0.57과 0.61으로 차이가 있다.

그럼 MSE 함수로 loss 값을 구해야된다. 얻기 바라는 값을 $t$, 실제로 나온 값을 $y$라고 할때 loss는 다음 과 같다.

$$
E = \frac{1}{2} \sum (t_i - y_i)^2
$$

결과랑 원하는 값이 얼마나 차이나는 지 구하고 그 값들의 평균을 내는 것 이다. 학습 시킨다는 것은 loss 값을 0에 근사 시킨다는 것 이다. 여기서 나온 loss값을 Backpropagation하면 된다.

```python
def MSE(targets, values):
    if not isinstance(values, list):
        return False

    result = 0
    for target, value in zip(targets, values):
        result += 0.5 * (target - value) ** 2

    return result

print(MSE([0.2, 0.7], [0.57, 0.61]))  # 결과: 0.0725
```

이제 구한 loss 값으로 역전파를 진행해보자.

## Back Propagation

![backprop backward](/assets/images/blog/image.png)


0.4로 할당되어 있는 가중치 $w_{10}^{1}$의 값을 업데이트할거다. 그러려면 $w_{10}^{1}$의 전체 에러인 $E$에 얼마나 영향을 미쳤는지, 즉 기여도를 구해한다. 이때 Chain rule이 사용된다.

$E$에 대해 $w_{10}^{1}$이 얼마나 영향을 미쳤는지 기여도 식으로 풀면 아래와 같다.

$$
\frac{\partial E}{\partial w_{10}^{1}} = \frac{\partial E}{\partial a_{20}} \frac{\partial a_{20}}{\partial z_{20}} \frac{\partial z_{20}}{\partial w_{10}^{1}}
$$

먼저 $\frac{\partial E}{\partial a_{20}}$부터 풀어보면, 원래 구한 $E$는 아래와 같은 식이다.

$$
E = \frac{1}{2} ((t_1 - a_{20})^2 + (t_2 - a_{21})^2)
$$

여기서 $a_{20} = y_1, a_{21} = y_2$ 이기 때문에 치환함, 하지만  $\frac{\partial E}{\partial a_{20}}$는 편미분 식으로 지금 구하려는 값과 상관 없는 $a_{21}$은 0으로 생각하고 풀면 된다.

$$
\frac{\partial E}{\partial a_{20}} = (t_{1} - a_{20}) \times -1 + 0 = (0.2 - 0.57) \times -1 = 0.37
$$

이 계산이 의미하는 것은 $E$에 대하여 $a_{20}$, 즉 $y_1$이 0.37만큼 기여한 것을 알 수 있다. 이런 식으로 계속해보자

$$
\frac{\partial a_{20}}{\partial z_{20}} = a_{20} \times (1 - a_{20}) = 0.57 \times (1 - 0.57) = 0.25
$$

$$
\frac{\partial z_{20}}{\partial w_{10}^{1}} = a_{10} + 0 = 0.54
$$

$$
\frac{\partial E}{\partial w_{10}^{1}} = 0.37 \times 0.25 \times 0.54 = 0.049
$$

최종적으로 $E$에 $w_{10}^{1}$가 기여한 값은 0.049이라는 값을 계산했다. 이 값을 학습식에 넣으면 $w_{10}^{1}$값을 업데이트할 수 있다.

이때 값을 얼마나 건너 뛸 것 이냐 또 얼마나 빨리 학습할 것 이냐를 정하는게 Learning Rate로 하이퍼 파라미터이다. 0.3으로 잡고 하겠다.

$$
w_{10}^{1+} = w_{10}^{1} - (L \times \frac{\partial E}{\partial w_{10}^{1}}) = 0.4 - (0.3 \times 0.049) = 0.3853
$$

이렇게 해서 새로운 $w_{10}^{1}$의 값을 구했다. 이렇게 다른 $w$값들도 구할 수 있다.
