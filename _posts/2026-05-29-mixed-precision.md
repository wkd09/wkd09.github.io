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

오늘은 GPU resource를 효율적으로 사용하는 기법중 하나인 Mixed Precision에 대해 알아 볼 것이다.
이번에도 단순한 개념이 아닌,
어떤 병목으로 인해 나온 방법론인지,
trade-off는 어떻게 되는지는 알아 보겠다.
Mixed Precision을 알아 보기 전에 부동 소수점부터 복습해보겠다.
## Floationg Point Fomat
![Mixed Precision image 1](https://velog.velcdn.com/images/junmin0413/post/6ae453b5-0c3b-4b8b-ad88-7c19aaa2457b/image.png)
• BF16: 총 16bits이며, 지수범위가 8bits, 가수범위가 7bits
• FP16: 총 16bits이며, 지수범위가 5bits, 가수범위가 10bits
• FP32: 총 32bits이며, 지수범위가 8bits, 가수범위가 23bits
지수: 숫자의 크기, 표현 가능한 최대/최소 값
가수: 숫자의 정밀도, 값의 정확도
즉, 표현 범위에 있어선 FP16 < BF16 < FP32이지만,
정밀도에 있어선 BF16 < FP16 < FP32이다.
그래서 모델 학습 시 항상 bf16이 fp16보다 좋다고 얘기할수만은 없다.
최근 학습에선 fp16를, 추론에서 양자화 8bit, 4bit를 많이 사용하는 것 같다.
fp16방식을 이용해 학습을 하면 저장공간도 아끼고, 학습 시간도 줄어든다.
하지만 정밀도가 매우 떨어진다. 따라 gradient가 너무 큰 경우, 너무 작은 경우 오차가 발생하게 되고 이 오차들이
누적되어 결국 학습이 잘 진행되지 않는다.
![Mixed Precision image 2](https://velog.velcdn.com/images/junmin0413/post/03e8c4f7-d871-4e1a-9e04-868f86e6a3e6/image.png)
그리고 fp32로 학습하면 배치 사이즈도 크게 늘릴 수 없고, 메모리 용량을 많이 차지해 메모리 통신 시간이 많이 걸린다는
단점이 존재한다.
이러한 문제들을 해결하기 위해 Mixed Precision이 나왔다.
## Mixed Precision
메모리와 정밀도의 trade-off 문제를 해결하기 나온 Mixed Precision.
## 방법론
![Mixed Precision image 3](https://velog.velcdn.com/images/junmin0413/post/53ad5f17-5197-4a76-ba64-8fd736945a04/image.png)
1. FP32 wheight에 대한 FP16 copy weight을 만든다. (FP16은 forward, backward에서 사용)
2. FP16 copy weight로 계산된 forward pass 진행
3. forward pass로 계산된 FP16 prediction 값을 FP32로 casting(타입변환)한다.
4. FP32 prediction을 이용해 FP32 loss를 계산하고 여기에 scaling factor S를 곱한다.
5. scaled FP32 loss를 FP16를 casting(타입변환)한다.
6. scaled FP16 loss를 이용해 backward propagation을 진행하고, gradient 계산.
7. FP16 gradient를 FP32로 casting하고, 이를 scaling factor S로 다시 나눈다. (chain rule로 인해 모든 gradient는
같은 크기로 scaling된 상태)
8. FP32 gradient를 이용해 FP32 weight를 update한다.
정리하면 FP 32 weight는 저장하고
FP 16 copy weight를 만들어 이를 이용해 forward/backward pass를 진행하고,
FP 16 copy weight으로 얻은 gradient를 이용해 FP32 weight를 update한다.
즉, 연산할땐 FP16으로 메모리 줄이고 update할땐 FP32로 정밀도를 높인다.
![Mixed Precision image 4](https://velog.velcdn.com/images/junmin0413/post/41acfc22-acba-4b21-8c0a-6cbf3f75c6ab/image.png)
계속 나온 scaling factor S는 뭐고 어떻게 정할까?
loss를 키워(scaling), under flow를 막기 위해 곱하는 것으로
그럼 어떻게 정하지?
-> 논문에선 단순히 경험적 값을 선택하거나,
gradient의 통계화가 가능한 경우 gradient의 maximum absolute value가 65,504가 되도록 맞춰 주면 된다고
한다. 하지만 scaling factor이 크다고 나쁜건 아니지만, overflow가 일어나지 않도록 주의!!
## 실험
그럼 Nvidia의 실험한 속도, 성능 표를 보자
![Mixed Precision image 5](https://velog.velcdn.com/images/junmin0413/post/f22a078b-80ed-4aff-b3a5-6255bf1b1129/image.png)
본 실험은 Translation, Speech recognition, Language modeling 등에 대해서 실험을 진행했는데, 속도가 매우
빨라지는 것을 확인 가능.
속도가 2~4.9배가 빨라지는 것을 볼 수 있다.
![Mixed Precision image 6](https://velog.velcdn.com/images/junmin0413/post/24dfb2e7-791d-4f82-a7fc-4c976d807257/image.png)
성능을 보아도, FP32이랑 거의 차이없거나 더 높은 성능 볼 수 있다.(성능 상승 이유:batch size 증강로 인해 noisy한
gradient가 없어지는 효과일수도?)
## 실습
Pytorch를 통해 실습해보자
```python
use_amp = True
net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
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
- <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.htmlhttps://bo-10000.tistory.com/32https://yjoonjang.medium.com/mixed-precision-training%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-mp-amp-torch-cuda-amp-15c99488ed34>
