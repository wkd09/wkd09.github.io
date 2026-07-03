---
title: "Data Parallelism과 DDP 정리"
date: 2026-05-29 00:10:00 +0900
last_modified_at: 2026-07-04 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - Training
  - GPU
  - Distributed Training
source: "Velog PDF - Data Parallelism & DDP"
---

저번에는 모델을 나누어 올리는 Model Parallelism을 알아보았다. 이번에는 데이터를 나누어 GPU에 올리는 Data Parallelism을 정리한다.

이번에도 단순한 개념 설명이 아니라, 각 방식이 어떤 병목을 해결하기 위해 나왔는지, 실제 학습 환경에서 어떤 trade-off를 가지는지를 중심으로 살펴본다.

먼저 Data Parallelism부터 알아보자.

## Data Parallelism(DP)

DP는 single process / multi-thread 방식으로, single node에서만 동작한다.

DP를 이용해 학습하는 과정은 다음과 같다.

![Data Parallelism & DDP image 1](https://velog.velcdn.com/images/junmin0413/post/b9c15af8-3993-4fe1-933c-310ac17bb473/image.png)

## Forward

![Data Parallelism & DDP image 2](https://velog.velcdn.com/images/junmin0413/post/6101831f-5ca0-45ac-9806-97e59f0286dc/image.png)

## Backward

Forward 과정:

1. 데이터가 mini batch로 분할되어 각 GPU에 할당된다(scatter).
2. 모든 GPU에 모델의 복사본을 올린다. 즉 모델 파라미터를 broadcast한다.
3. 하나의 GPU에 있는 모델만 Main Model로 사용한다.
4. 각 GPU는 할당된 데이터를 이용해 독립적으로 forward pass를 수행한다.
5. Forward 이후, 모든 GPU의 출력이 Main Model이 있는 GPU로 수집되어 한 번에 loss가 계산된다(gather).

Backward 과정:

1. Main Model에서 계산된 loss에 대한 output gradient를 다른 GPU로 뿌린다(scatter).
2. 각 GPU에서 backward pass가 수행되어 gradient가 계산된다.
3. gradient는 다시 Main Model이 속한 GPU로 수집된다(gather).
4. optimizer step을 통해 parameter가 update된다.
5. 다시 모든 GPU에 모델 복사본을 broadcast하고 과정을 반복한다.

## 장단점

## 장점

- n개의 GPU에 동일한 모델을 올려 학습하므로 학습 속도를 높일 수 있다.
- 같은 이유로 검증 및 예측 속도도 높일 수 있다.
- 여러 GPU에 batch를 분할해 학습하므로 batch size를 크게 구성할 수 있다.

## 단점

- Python에서는 multi-thread 방식이 성능을 저하시킬 수 있다.
- step마다 각 GPU에 업데이트된 모델을 broadcast해야 한다.
- Main Model에 gather가 몰리기 때문에 병목이 생길 수 있다.

이런 단점을 극복하기 위해 나온 방법이 DDP이다.

## Distributed Data Parallelism(DDP)

DDP는 single-node/multi-node, multi-GPU에서 동작하는 multi-process 방식이다.

DDP의 학습 과정은 다음과 같다.

![Data Parallelism & DDP image 3](https://velog.velcdn.com/images/junmin0413/post/322a3770-fc5f-4b06-ac0e-c78cc3868197/image.png)

1. DDP 방식에선 모든 GPU에 모델의 복사본을 broadcast하고, 모든 모델이 Main Model이다.
2. 데이터가 mini batch로 분할되어 각 GPU에 할당된다.
3. 각 GPU는 할당된 데이터를 이용해 독립적으로 forward pass를 수행한다. 여기까지는 DP와 동일하다.
4. Forward 후, 독립적으로 backward pass를 수행하여 gradient를 계산한다.
5. Gradient를 AllReduce 방식으로 집계한다. AllReduce는 각 GPU에서 계산된 gradient의 평균을 구하는 방식이다.
6. AllReduce를 통해 계산된 gradient를 사용하여 model parameter를 업데이트한다.

그런데 AllReduce도 통신 비용이 있다. 각 GPU로부터 gradient 값을 받아야 가중치를 업데이트할 수 있기 때문이다.

대표적으로 `Reduce -> Broadcast` 방식은 gradient 값을 각 GPU로부터 한 GPU에 모아 합산한 뒤, 다시 모든 GPU로 보내야 한다. 이 방식은 특정 GPU에 부담이 커진다는 단점이 있다.

![Data Parallelism & DDP image 4](https://velog.velcdn.com/images/junmin0413/post/bcd90fc7-53a3-4d71-bb0f-580bf9cd06e9/image.png)

이러한 단점을 해결하기 위해 나온 방식이 Ring AllReduce다.

## Ring All-reduce

Baidu에서 Ring AllReduce라는 방법을 제안했다.

![Data Parallelism & DDP image 5](https://velog.velcdn.com/images/junmin0413/post/4ba8c1c4-38a6-4cf1-b13f-0531c1321db4/image.png)

이름 그대로 각 GPU를 ring 형태로 돌면서 gradient를 나누어 합치고, 다시 돌면서 합쳐진 gradient 값을 공유하는 방식이다. 특정 device 하나로 모든 값이 모이지 않기 때문에 하나의 GPU에 부담이 집중되지 않는다.

DDP를 이용해 학습하면 Main Model에게 forward pass를 넘겨 계산하고 다시 다른 GPU에게 뿌리는 전송/집계 병목을 줄일 수 있다.

## PyTorch DDP

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)

    model = nn.Linear(1024, 512).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)

        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(rank)
            targets = targets.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()  # 여기서 자동으로 gradient AllReduce 수행
            optimizer.step()

    cleanup()


# 실행: torchrun --nproc_per_node=4 train_script.py
```

## 참고 자료

- <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>
- <https://sonstory.tistory.com/123>
