---
title: "Mixed Precision 정리: 메모리와 정밀도의 trade-off 다루기"
date: 2026-05-29 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - Training
  - GPU
  - Mixed Precision
source: "Velog PDF - Mixed Precision"
---

Mixed Precision은 GPU resource를 효율적으로 쓰기 위한 학습 기법이다.

핵심은 모든 계산을 FP32로 하지 않고, 가능한 연산은 FP16으로 처리하면서도 weight update처럼 정밀도가 중요한 부분은 FP32를 유지하는 것이다. 즉, 메모리 사용량과 연산 속도는 FP16의 장점을 가져가고, 학습 안정성은 FP32로 보완한다.

## Floating Point Format

학습에서 자주 보는 floating point format은 FP32, FP16, BF16이다.

- BF16: 총 16bit, 지수 8bit, 가수 7bit
- FP16: 총 16bit, 지수 5bit, 가수 10bit
- FP32: 총 32bit, 지수 8bit, 가수 23bit

지수는 표현 가능한 값의 범위에 가깝고, 가수는 값의 정밀도에 가깝다.

표현 범위만 보면 `FP16 < BF16 < FP32`이고, 정밀도는 `BF16 < FP16 < FP32`이다. 그래서 항상 BF16이 FP16보다 좋다고 말할 수는 없다. 상황에 따라 필요한 것은 범위일 수도 있고, 정밀도일 수도 있다.

## FP16만 쓰면 생기는 문제

FP16을 사용하면 저장 공간과 학습 시간을 줄일 수 있다. 하지만 정밀도가 낮아 gradient가 너무 작거나 큰 경우 문제가 생긴다.

작은 gradient는 underflow로 0에 가까워질 수 있고, 큰 값은 overflow를 만들 수 있다. 이런 오차가 누적되면 학습이 제대로 진행되지 않는다.

반대로 FP32만 사용하면 정밀도는 좋지만 메모리를 많이 쓰고, batch size를 크게 가져가기 어렵다. GPU 메모리 사용량이 커지면 메모리 통신 비용도 늘어난다.

Mixed Precision은 이 trade-off를 줄이기 위해 나왔다.

## Mixed Precision의 방식

Mixed Precision 학습 흐름은 다음과 같다.

1. FP32 weight를 master weight로 저장한다.
2. FP32 weight의 FP16 copy weight를 만든다.
3. FP16 copy weight로 forward pass를 수행한다.
4. FP16 prediction을 FP32로 casting한다.
5. FP32 prediction으로 loss를 계산하고 loss scaling factor `S`를 곱한다.
6. scaled loss를 FP16으로 casting한 뒤 backward propagation을 수행한다.
7. 계산된 FP16 gradient를 FP32로 casting한다.
8. gradient를 scaling factor `S`로 다시 나눈다.
9. FP32 gradient로 FP32 master weight를 update한다.

정리하면 연산할 때는 FP16으로 메모리와 시간을 줄이고, update할 때는 FP32로 정밀도를 유지한다.

## Loss Scaling

계속 등장하는 scaling factor `S`는 loss를 키워서 gradient underflow를 막기 위한 값이다.

FP16은 표현 가능한 값의 범위와 정밀도가 제한적이기 때문에 작은 gradient가 0으로 사라질 수 있다. loss에 `S`를 곱하면 backward 과정에서 gradient도 같은 비율로 커진다. 이후 update 전에 다시 `S`로 나누면 원래 scale로 복원된다.

단, `S`를 너무 크게 잡으면 overflow가 생길 수 있다. 논문에서는 경험적인 값을 선택하거나, gradient 통계를 보고 maximum absolute value가 FP16 표현 범위 안에 들어오도록 조정하는 방식을 이야기한다.

## PyTorch AMP 예시

PyTorch에서는 `autocast`와 `GradScaler`를 사용해 Mixed Precision을 적용할 수 있다.

```python
use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = net(input)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
```

## 정리

Mixed Precision의 핵심은 다음과 같다.

- FP16으로 forward/backward 연산 비용을 줄인다.
- FP32 master weight로 update 정밀도를 유지한다.
- loss scaling으로 FP16 gradient underflow를 완화한다.
- 메모리를 줄여 더 큰 batch size를 사용할 수 있다.

학습 성능을 올릴 때 단순히 모델 구조만 보는 것이 아니라, 숫자 표현 방식과 GPU 메모리 병목도 함께 봐야 한다.

## 참고 자료

- <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- <https://docs.pytorch.org/docs/stable/amp.html>
