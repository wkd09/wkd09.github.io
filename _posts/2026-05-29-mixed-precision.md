---
title: "Mixed Precision 정리: 메모리와 정밀도의 trade-off 다루기"
date: 2026-05-29 00:00:00 +0900
last_modified_at: 2026-07-04 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - Training
  - GPU
  - Mixed Precision
source: "Velog PDF - Mixed Precision"
---

Mixed Precision은 GPU resource를 효율적으로 사용하기 위한 대표적인 학습 기법이다.

단순히 개념만 보는 것이 아니라, 어떤 병목 때문에 이 방법이 나왔고 메모리와 정밀도 사이에서 어떤 trade-off가 생기는지 살펴보자. Mixed Precision을 이해하기 전에 먼저 부동소수점 표현부터 복습한다.

## Floating Point Format

![Mixed Precision image 1](https://velog.velcdn.com/images/junmin0413/post/6ae453b5-0c3b-4b8b-ad88-7c19aaa2457b/image.png)

- BF16: 총 16bits이며, 지수 범위가 8bits, 가수 범위가 7bits
- FP16: 총 16bits이며, 지수 범위가 5bits, 가수 범위가 10bits
- FP32: 총 32bits이며, 지수 범위가 8bits, 가수 범위가 23bits

지수는 숫자의 크기와 표현 가능한 최대/최소값에 영향을 주고, 가수는 숫자의 정밀도에 영향을 준다.

즉 표현 범위는 대체로 `FP16 < BF16 < FP32`이고, 정밀도는 `BF16 < FP16 < FP32`이다. 그래서 모델 학습 시 항상 BF16이 FP16보다 좋다고 말할 수는 없다.

FP16 방식으로 학습하면 저장 공간을 아끼고 학습 시간도 줄일 수 있다. 하지만 정밀도가 낮아 gradient가 너무 크거나 작은 경우 오차가 발생할 수 있고, 이 오차가 누적되면 학습이 불안정해질 수 있다.

![Mixed Precision image 2](https://velog.velcdn.com/images/junmin0413/post/03e8c4f7-d871-4e1a-9e04-868f86e6a3e6/image.png)

반대로 FP32로만 학습하면 배치 사이즈를 크게 늘리기 어렵고, 메모리를 많이 차지해 메모리 통신 시간도 커진다. 이러한 문제를 줄이기 위해 Mixed Precision이 사용된다.

## Mixed Precision

Mixed Precision은 메모리 사용량과 정밀도 사이의 trade-off를 다루기 위한 방법이다. 핵심은 모든 값을 낮은 precision으로 처리하는 것이 아니라, 필요한 부분은 FP32로 유지하고 계산량이 큰 부분은 FP16/BF16을 활용하는 것이다.

## 방법론

![Mixed Precision image 3](https://velog.velcdn.com/images/junmin0413/post/53ad5f17-5197-4a76-ba64-8fd736945a04/image.png)

1. FP32 weight에 대한 FP16 copy weight를 만든다. FP16은 forward/backward에서 사용한다.
2. FP16 copy weight로 forward pass를 진행한다.
3. forward pass로 계산된 FP16 prediction 값을 FP32로 casting한다.
4. FP32 prediction을 이용해 FP32 loss를 계산하고 여기에 scaling factor S를 곱한다.
5. scaled FP32 loss를 FP16으로 casting한다.
6. scaled FP16 loss로 backward propagation을 진행하고 gradient를 계산한다.
7. FP16 gradient를 FP32로 casting하고, 이를 scaling factor S로 다시 나눈다. chain rule로 인해 모든 gradient는 같은 크기로 scaling된 상태다.
8. FP32 gradient를 이용해 FP32 weight를 update한다.

정리하면 FP32 weight는 저장해두고, FP16 copy weight로 forward/backward pass를 진행한 뒤, FP16에서 얻은 gradient를 이용해 FP32 weight를 update한다. 즉 연산할 때는 FP16으로 메모리를 줄이고, update할 때는 FP32로 정밀도를 높인다.

![Mixed Precision image 4](https://velog.velcdn.com/images/junmin0413/post/41acfc22-acba-4b21-8c0a-6cbf3f75c6ab/image.png)

계속 등장한 scaling factor S는 무엇이고 어떻게 정할까?

loss scaling은 loss를 키워서 underflow를 막기 위해 사용하는 방법이다. 논문에서는 경험적인 값을 선택하거나, gradient 통계를 사용할 수 있는 경우 gradient의 maximum absolute value가 65,504 근처가 되도록 맞추는 방법을 설명한다. 다만 scaling factor가 너무 크면 overflow가 발생할 수 있으므로 주의해야 한다.

## 실험

NVIDIA의 실험 결과를 보자.

![Mixed Precision image 5](https://velog.velcdn.com/images/junmin0413/post/f22a078b-80ed-4aff-b3a5-6255bf1b1129/image.png)

Translation, speech recognition, language modeling 등에서 실험을 진행했고, 속도가 2~4.9배 빨라지는 것을 볼 수 있다.

![Mixed Precision image 6](https://velog.velcdn.com/images/junmin0413/post/24dfb2e7-791d-4f82-a7fc-4c976d807257/image.png)

성능도 FP32와 거의 차이가 없거나 더 높은 경우를 볼 수 있다. 성능 상승은 batch size 증가로 noisy한 gradient가 줄어드는 효과와 관련이 있을 수 있다.

## 실습

PyTorch를 통해 실습해보자.

```python
use_amp = True
net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

start_timer()

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

end_timer_and_print("Mixed precision:")
```

## 참고 자료

- <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- <https://bo-10000.tistory.com/32>
- <https://yjoonjang.medium.com/mixed-precision-training%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-mp-amp-torch-cuda-amp-15c99488ed34>
