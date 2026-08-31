---
title: "Mixed Precision 정리: 메모리와 정밀도의 trade-off 다루기"
date: 2026-05-29 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - Training
  - GPU
  - Mixed Precision
source: "Velog PDF - Mixed Precision"
---

Mixed Precision은 model training에서 낮은 precision과 높은 precision을 함께 사용하는 방법이다.

모든 값을 FP32로 계산하면 안정적이지만 memory 사용량과 memory traffic이 커진다. 반대로 모든 값을 FP16으로 바꾸면 빠르고 memory를 적게 사용하지만 작은 gradient가 0이 되는 underflow나 큰 값이 표현 범위를 넘는 overflow가 생길 수 있다.

핵심은 하나의 dtype만 고집하지 않는 것이다.

> 큰 matrix operation은 FP16 또는 BF16으로 계산하고, 정밀도가 필요한 값은 FP32로 유지한다.

이 글에서는 FP16과 BF16의 차이, FP32 master weight와 loss scaling, PyTorch AMP의 실행 흐름을 정리한다.

## Floating Point Format

![Mixed Precision image 1](https://velog.velcdn.com/images/junmin0413/post/6ae453b5-0c3b-4b8b-ad88-7c19aaa2457b/image.png)

| Format | Sign | Exponent | Fraction | 특징 |
| --- | ---: | ---: | ---: | --- |
| FP16 | 1 bit | 5 bits | 10 bits | BF16보다 정밀하지만 표현 범위가 좁음 |
| BF16 | 1 bit | 8 bits | 7 bits | FP32와 exponent 범위가 같아 overflow에 강함 |
| FP32 | 1 bit | 8 bits | 23 bits | 넓은 범위와 높은 정밀도, memory 사용량이 큼 |

지수는 숫자의 크기와 표현 가능한 최대/최소값에 영향을 주고, 가수는 숫자의 정밀도에 영향을 준다.

즉 표현 범위는 대체로 `FP16 < BF16 ≈ FP32`이고, fraction 정밀도는 `BF16 < FP16 < FP32`다. BF16은 FP16보다 표현 범위가 넓지만 fraction bit는 적다.

낮은 precision을 사용하면 tensor 크기와 memory traffic을 줄일 수 있고, 지원되는 GPU에서는 Tensor Core를 활용해 matrix multiplication도 빨라질 수 있다. 문제는 gradient가 너무 작거나 큰 경우다. 이 오차가 누적되면 학습이 불안정해질 수 있다.

![Mixed Precision image 2](https://velog.velcdn.com/images/junmin0413/post/03e8c4f7-d871-4e1a-9e04-868f86e6a3e6/image.png)

반대로 FP32만 사용하면 같은 GPU에 넣을 수 있는 batch size가 작아지고 memory read/write도 늘어난다. Mixed Precision은 이 두 문제 사이에서 필요한 precision만 남긴다.

## Mixed Precision

Mixed Precision에서는 operation마다 적절한 dtype을 선택한다.

```text
Matrix multiplication / convolution -> FP16 또는 BF16
일부 reduction과 loss             -> FP32
Master weight와 optimizer state    -> FP32로 유지할 수 있음
```

PyTorch의 `autocast`는 operation별 dtype 선택을 자동으로 처리한다. 무조건 모든 tensor를 FP16으로 cast하는 방식과 다르다.

## 방법론

![Mixed Precision image 3](https://velog.velcdn.com/images/junmin0413/post/53ad5f17-5197-4a76-ba64-8fd736945a04/image.png)

논문에서 설명하는 FP16 mixed precision의 기본 흐름은 다음과 같다.

1. FP32 master weight를 유지하고 forward/backward용 FP16 copy를 만든다.
2. FP16 weight로 forward pass를 계산한다.
3. Loss를 FP32에서 계산하고 scaling factor $S$를 곱한다.
4. Scaled loss로 backward를 수행해 작은 gradient가 FP16 범위에서 사라지지 않게 한다.
5. Gradient를 FP32로 바꾸고 $S$로 나눠 원래 크기로 복원한다.
6. FP32 master weight를 update한다.

즉 계산량이 큰 forward/backward는 FP16으로 처리하고 parameter update에는 FP32 precision을 남긴다.

![Mixed Precision image 4](https://velog.velcdn.com/images/junmin0413/post/41acfc22-acba-4b21-8c0a-6cbf3f75c6ab/image.png)

### Loss Scaling

FP16에서 아주 작은 gradient는 표현 범위 아래로 내려가 0이 될 수 있다. Loss에 $S$를 곱하면 chain rule에 따라 모든 gradient도 $S$배 커진다. Update 직전에 다시 $S$로 나누면 수학적으로 같은 gradient를 얻으면서 underflow를 줄일 수 있다.

문제는 $S$가 너무 크면 overflow가 생긴다는 점이다. PyTorch `GradScaler`는 overflow를 감지해 해당 update를 건너뛰고 scale을 낮춘다. 안정적인 step이 이어지면 scale을 다시 높인다.

BF16은 FP32와 exponent 범위가 같아 FP16보다 loss scaling이 덜 필요한 경우가 많다.

## 실험

NVIDIA의 mixed precision 연구에서는 translation, speech recognition, language modeling 등 여러 task를 비교했다.

![Mixed Precision image 5](https://velog.velcdn.com/images/junmin0413/post/f22a078b-80ed-4aff-b3a5-6255bf1b1129/image.png)

논문 기준으로 일부 workload에서 FP32 대비 2~4.9배 speedup을 기록했다.

![Mixed Precision image 6](https://velog.velcdn.com/images/junmin0413/post/24dfb2e7-791d-4f82-a7fc-4c976d807257/image.png)

Model quality도 FP32와 비슷하게 유지됐다. 다만 이 배수는 당시 GPU와 workload에서 나온 결과다. 현재 환경의 speedup은 GPU architecture, Tensor Core 지원, model shape와 memory bottleneck에 따라 달라진다.

## 실습

PyTorch에서는 `autocast`와 `GradScaler`를 함께 사용할 수 있다.

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

`autocast`는 operation별 dtype을 선택하고, `GradScaler`는 FP16 gradient underflow를 줄인다. BF16을 사용할 때는 일반적으로 `GradScaler` 없이 `autocast(dtype=torch.bfloat16)`만 사용하는 경우가 많다.

## 내가 이해한 핵심

Mixed Precision은 모든 값을 낮은 precision으로 바꾸는 방법이 아니다.

```text
낮은 precision
-> matrix operation과 memory traffic을 줄임

FP32 유지
-> 작은 update와 optimizer state의 안정성 확보

Loss scaling
-> FP16에서 작은 gradient가 0이 되는 문제 완화
```

핵심은 속도와 정확도 중 하나를 포기하는 것이 아니라, 각 operation에 필요한 precision만 사용하는 것이다. 실제 적용에서는 FP16과 BF16 중 무엇을 쓰는지, overflow로 update가 skip되는지, 최종 loss가 FP32 baseline과 비슷하게 수렴하는지를 함께 확인해야 한다.

## 참고 자료

- <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- <https://bo-10000.tistory.com/32>
- <https://yjoonjang.medium.com/mixed-precision-training%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-mp-amp-torch-cuda-amp-15c99488ed34>
