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

Model Parallelism이 모델을 나누어 GPU에 올리는 방식이라면, Data Parallelism은 데이터를 나누어 여러 GPU에서 같은 모델을 학습시키는 방식이다.

목표는 단순하다. 같은 모델 복사본을 여러 GPU에 올리고, mini batch를 나눠 처리해 학습 속도와 batch size를 늘리는 것이다.

## DataParallelism

PyTorch의 `DataParallel`은 single process, multi-thread 방식으로 동작한다. 보통 single node multi GPU에서 사용한다.

학습 흐름은 다음과 같다.

1. mini batch를 여러 GPU에 나눈다. 이것을 scatter라고 한다.
2. 모든 GPU에 모델 복사본을 올린다.
3. 각 GPU는 할당된 데이터로 forward pass를 수행한다.
4. 각 GPU의 output을 main GPU로 gather한다.
5. main GPU에서 loss를 계산한다.
6. loss gradient를 다시 각 GPU로 scatter한다.
7. 각 GPU에서 backward pass로 gradient를 계산한다.
8. gradient를 main GPU로 gather한다.
9. main GPU에서 parameter를 update한다.
10. update된 parameter를 다시 모든 GPU에 broadcast한다.

## DataParallel의 장단점

장점은 직관적이다.

- 여러 GPU로 학습 속도를 높일 수 있다.
- batch size를 크게 가져갈 수 있다.
- 검증과 예측 속도도 개선될 수 있다.

하지만 단점도 분명하다.

- Python multi-thread 방식의 한계가 있다.
- main GPU에 gather가 몰린다.
- 매 step마다 update된 모델을 다시 broadcast해야 한다.
- main GPU가 병목이 되기 쉽다.

이 한계를 줄이기 위해 주로 사용하는 방식이 DDP다.

## DistributedDataParallel

DDP는 single node뿐 아니라 multi node multi GPU에서도 사용할 수 있는 multi process 방식이다.

DDP에서는 각 GPU마다 별도 process가 있고, 모든 GPU의 모델이 main model처럼 동작한다. 특정 GPU 하나에 output과 gradient가 몰리지 않는다.

흐름은 다음과 같다.

1. 모든 GPU에 동일한 모델 복사본을 둔다.
2. mini batch를 GPU별로 나눈다.
3. 각 GPU가 독립적으로 forward pass를 수행한다.
4. 각 GPU가 독립적으로 backward pass를 수행한다.
5. gradient를 all-reduce로 평균낸다.
6. 평균 gradient로 각 GPU의 model parameter를 update한다.

즉, DDP는 output을 한 GPU에 모아 loss를 계산하는 구조가 아니라, 각 process가 자기 몫의 데이터를 처리하고 gradient만 동기화한다.

## All-Reduce와 통신 비용

DDP의 핵심은 gradient all-reduce다.

각 GPU에서 계산한 gradient를 평균내어 모든 GPU가 같은 update를 수행하게 만든다. 하지만 gradient를 주고받는 통신 비용은 작지 않다.

단순한 reduce -> broadcast 방식은 한 GPU에 gradient를 모아 합산한 뒤 다시 모든 GPU에 뿌린다. 이 경우 특정 GPU에 부담이 몰린다.

이를 개선한 방식이 Ring All-Reduce다.

## Ring All-Reduce

Ring All-Reduce는 GPU를 ring 형태로 연결해 gradient 조각을 순차적으로 주고받으며 합산하고, 다시 한 바퀴 돌며 결과를 공유한다.

장점은 다음과 같다.

- 특정 GPU에 통신이 몰리지 않는다.
- 모든 GPU가 통신에 고르게 참여한다.
- 큰 gradient 동기화에서 효율적이다.

DDP를 사용하면 `DataParallel`에서 main GPU로 forward output과 gradient를 모으는 병목을 줄일 수 있다. 대신 gradient 동기화 통신 비용을 어떻게 줄이느냐가 중요해진다.

## PyTorch DDP 예시

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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
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
        for inputs, targets in dataloader:
            inputs = inputs.to(rank)
            targets = targets.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

    cleanup()
```

실행은 보통 다음처럼 한다.

```bash
torchrun --nproc_per_node=4 train_script.py
```

## 정리

Data Parallelism은 데이터를 나눠 여러 GPU에서 같은 모델을 학습하는 방식이다.

`DataParallel`은 구조가 단순하지만 main GPU 병목이 생기기 쉽다. DDP는 multi process 방식으로 각 GPU가 독립적으로 forward/backward를 수행하고 gradient만 all-reduce로 동기화한다.

분산 학습에서는 계산량뿐 아니라 통신량이 성능을 결정한다. GPU를 많이 붙인다고 항상 빨라지는 것이 아니라, gradient 동기화 비용을 어떻게 줄이는지가 중요하다.
