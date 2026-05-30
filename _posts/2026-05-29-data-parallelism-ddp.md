---
title: "Data Parallelism과 DDP 정리"
date: 2026-05-29 00:10:00 +0900
categories:
  - engineering
tags:
  - AI
  - Training
  - GPU
  - Distributed Training
source: "Velog PDF - Data Parallelism & DDP"
---

저번엔 모델을 나누어 올리는 Model Parallelism을 알아 보았는데 이번엔 데이터를 나누어 GPU에 올리는 Data
Parallelism을 작성할 것 이다.
이번에도 단순히 개념이 아닌,
각 방식이 어떤 병목을 해결하기 위해 나왔는지,
실제 서빙 환경에서 어떤 trade-off를 가지는지를 중심으로 살펴 볼 것이다.
먼저 DataParallelism에 대해서 알아보자
## DataParallelism(DP)
DP는 single process / multi-thread 방식으로, single node에서만 동작한다.
DP를 이용해 학습하는 과정은 다음과 같다.
![Data Parallelism & DDP image 1](https://velog.velcdn.com/images/junmin0413/post/b9c15af8-3993-4fe1-933c-310ac17bb473/image.png)
## Foward
![Data Parallelism & DDP image 2](https://velog.velcdn.com/images/junmin0413/post/6101831f-5ca0-45ac-9806-97e59f0286dc/image.png)
## Backward
--forward--
1. 데이터가 mini batch로 분할되어 각 GPU에 할당되어 각 GPU에 할당됨.(scatter)
2. 모든 GPU에 모델의 복사본을 넣고(모델의 파라미터를 복사, 즉 broadcast), 하나의 GPU에 있는 모델만 Main
Model로 사용
3. 각 GPU는 할당된 데이터를 이용해 독립적으로 forward pass를 수행
4. Forward 이후, 모든 GPU의 출력이 Main Model이 있는 GPU로 수집(Gather)되어 한번에 loss가 계산된다.
--backward--
5. Main Model에서 계산된 loss에 대한 output의 gradient를 다른 GPU 뿌린다.(Scatter)
6. 각 GPU에서 backward pass가 수행되어 gradient가 계산되고,
7. gradient는 다시 Main Model이 속한 GPU로 수집됨 (Gather)
8. step을 통해 parameter가 update 된다.
--forward--
9. 다시 모든 GPU에 모델의 복사본 broadcast하고, 1-8과정 반복.
## 장단점
## 장점
• n개의 GPU에 동일한 모델을 올려 학습함으로 학습 속도 증가
• 동일한 이유로 검증 및 예측 속도 증가
• 여러개의 GPU에 분할한 뒤 학습하여 batch size를 크게 구성할 수 있음
## 단점
• Python에선 Multi thread 방식이 성능을 저하 시킴
• step마다 각 GPU에 업데이트 된 모델을 broadcast해야됨
위 단점에서 나온 Multi thread 방식의 성능 저하 문제가 있음
그래서 단점을 극복하고자 나온 방법이 DDP이다.
## DistributedDataParallelism(DDP)
DDP는 single/multi-node & multi GPU에서 동작하는 multi process 모듈방식이다.
DDP의 학습 과정은 다음과 같다.
![Data Parallelism & DDP image 3](https://velog.velcdn.com/images/junmin0413/post/322a3770-fc5f-4b06-ac0e-c78cc3868197/image.png)
1. DDP 방식에선 모든 GPU에 모델의 복사본을 broadcast하고, 모든 모델이 Main Model이다.
2. 데이터가 mini batch로 분할되어 각 GPU에 할당
3. 각 GPU는 할당된 데이터를 이용해 독립적으로 forward pass 수행한다.
(여기까지 DP과 동일함)
4. Forward 후, 독립적으로 backward pass를 수행하여 gradient를 계산한다.
5. Gradient를 All-reduce 방식을 통해 집계. All-reduce 방식이란, 각 GPU에서 계산된 gradient의 평균을 구하는 것
6. 이렇게 All-reduce를 통해 계산된 gradient를 사용하여, model parameter를 업데이트한다.
그런데 All-reduce 방식도 복잡하다. 왜냐 각 GPU로 부터 gradient 값을 받아야 가중치가 업데이트 가능한데
gradient 값을 주고 받는 통신 비용이 만만치 않다.
대표적으로 Reduce -> Broadcast 방식은, gradient값을 각 GPU로부터 한 GPU에 모아 합산 값을 계산하고, 다시
하나의 GPU로 부터 모든 GPU에 보내야하는데, 이는 하나에 GPU에 부담이 커진다는 단점이 있다.
![Data Parallelism & DDP image 4](https://velog.velcdn.com/images/junmin0413/post/bcd90fc7-53a3-4d71-bb0f-580bf9cd06e9/image.png)
이러한 단점을 해결하기 위해 나온 것이
## Ring All-reduce
BAIDU에서 Ring All-reduce라는 방법 제안했다.
![Data Parallelism & DDP image 5](https://velog.velcdn.com/images/junmin0413/post/4ba8c1c4-38a6-4cf1-b13f-0531c1321db4/image.png)
이름 그대로, 각 GPU를 ring 형태로 돌면서 gradient를 하나하나 합치고, 또 돌면서 합쳐진 gradient 값을 보내는
방식으로, 특정 device로 모아지지 않아 하나의 GPU에 부담이 없고, 비효율적 연산을 수행하지 않는다.
DDP를 이용해 학습을 진행하면, Main Model에게 forward pass를 넘겨 주어, 계산하고, 다시 다른 GPU에게 뿌리는
전송, 집계 과정에서 발생하는 I/O-bound를 줄일 수 있다.
## Pytorch DPP
```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torhc
from torch.utils.data import DataLoader, DistributedSampler
def setup(rank, world_size):
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(rank)
def cleanup():
dist.destroy_process_group()
def train(rank, world_size):
setup(rank, world_size)
# 모델 생성 및 DDP 래핑
model = nn.Linear(1024, 512).to(rank)
ddp_model = DDP(model, device_ids=[rank])
# Dataset 및 DataLoader (DistributedSampler 사용 필수)
dataset = MyDataset()
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
for epoch in range(10):
sampler.set_epoch(epoch) # 매 epoch마다 호출하여 데이터 셔플 보장
for batch in dataloader:
inputs, targets = batch
inputs = inputs.to(rank)
targets = targets.to(rank)
optimizer.zero_grad()
outputs = ddp_model(inputs)
loss = loss_fn(outputs, targets)
loss.backward() # 여기서 자동으로 gradient AllReduce 수행
optimizer.step()
cleanup()
# 실행: torchrun --nproc_per_node=4 train_script.py
```
## 참고 자료
- <https://docs.pytorch.org/docs/stable/amp.htmlhttps://sonstory.tistory.com/123>
