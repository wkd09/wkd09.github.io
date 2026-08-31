---
title: "Pipeline Parallelism과 Tensor Parallelism 정리"
date: 2026-05-29 00:20:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - Training
  - Inference
  - Parallelism
source: "Velog PDF - Pipeline/Tensor Parallelism"
---

LLM이 GPU 한 장의 memory에 들어가지 않으면 model computation을 여러 GPU로 나눠야 한다. Pipeline Parallelism(PP)과 Tensor Parallelism(TP)은 모두 model을 나누지만 분할 기준이 다르다.

```text
Pipeline Parallelism : layer 묶음을 GPU별로 나눈다
Tensor Parallelism   : 한 layer의 matrix operation을 나눈다
```

핵심은 GPU 수가 아니라 병목이다. PP는 stage 사이의 bubble이 문제고, TP는 layer마다 발생하는 통신이 문제다. 이 글에서는 두 방식이 model을 어떻게 나누고 어떤 trade-off를 만드는지 정리한다.

## 1. Pipeline Parallelism (PP)

Pipeline Parallelism은 model을 layer 묶음으로 나눠 여러 GPU에 배치한다.

다음 상황에서 사용할 수 있다.

- 모델이 커서 단일 GPU 메모리에 올라가지 않을 때
- GPU 활용률과 처리량(throughput)을 높이고 싶을 때
- Tensor Parallelism 대비 통신 패턴을 단순화하고 싶을 때

문제는 stage가 항상 동시에 일하지 않는다는 점이다.

1. 스테이지 간 타이밍 불일치로 bubble time이 생긴다.
2. 일부 GPU가 idle 상태가 되어 utilization이 떨어진다.
3. bubble을 줄이려 micro-batch를 늘리면 activation memory가 증가한다.

즉 PP는 model memory를 나눌 수 있지만, pipeline을 채우고 비우는 동안 GPU가 쉬는 bubble을 함께 관리해야 한다.

### 1.1 GPipe

GPipe의 흐름은 다음과 같다.

- batch를 micro-batch로 쪼갠다.
- GPU 스테이지를 파이프라인으로 연결해 병렬 처리한다.

Micro-batch가 pipeline을 연속해서 지나가면 서로 다른 stage가 동시에 다른 batch를 처리할 수 있다.

장점은 다음과 같다.

- micro-batch 수가 늘어 GPU idle이 감소한다.
- pipeline 병렬성이 올라 throughput이 향상된다.

하지만 첫 micro-batch가 마지막 stage에 도착하기 전과 마지막 작업이 끝나는 구간에는 GPU가 쉰다.

- 파이프라인 마지막 flush 구간에서 idle이 발생한다.

Micro-batch를 늘리면 bubble 비율은 줄지만 activation을 더 오래 보관해야 하므로 memory 사용량이 늘 수 있다.

### 1.2 PipeDream (1F1B)

PipeDream의 1F1B schedule은 GPipe의 idle 구간을 줄이기 위한 방식이다.

```text
1 Forward -> 1 Backward -> 1 Forward -> 1 Backward
```

각 stage가 forward와 backward를 번갈아 수행해 쉬는 시간을 줄인다.

- `1 Forward -> 1 Backward`를 반복한다.
- 각 GPU가 쉬지 않고 연산을 이어간다.

- GPU utilization 증가
- pipeline flush 비용 감소

주의할 점도 있다.

- Forward와 Backward 사이에 파라미터 버전이 달라져 gradient inconsistency가 생길 수 있다.
- 이를 막기 위해 weight versioning이 필요하다.

즉 연산 효율을 높이는 대신 여러 weight version과 activation을 관리하는 부담이 생긴다.

### 1.3 PP 요약

| 방식 | 장점 | 핵심 부담 |
| --- | --- | --- |
| GPipe | schedule이 단순하고 memory 관리가 비교적 쉬움 | Pipeline bubble |
| PipeDream 1F1B | GPU idle과 flush 비용 감소 | Weight version과 activation 관리 |

## 2. Tensor Parallelism (TP)

Tensor Parallelism은 layer를 통째로 나누지 않는다. 하나의 layer 안에 있는 weight matrix와 matrix multiplication을 여러 GPU로 분할한다.

- PP: 레이어 기준 분할(세로)
- TP: 연산 기준 분할(가로)

대표적인 구현이 Megatron-LM이다.

### 2.1 Column Parallelism

Column Parallelism은 weight를 output dimension 방향으로 나눈다. 같은 input을 각 GPU가 받고 서로 다른 output block을 계산한다.

연산 흐름:
1. 동일한 input을 각 GPU에 전달한다.
2. 각 GPU가 자신이 가진 column weight로 부분 output을 계산한다.
3. 전체 output이 필요하면 AllGather로 결합한다.

### 2.2 Row Parallelism

Row Parallelism은 weight를 input dimension 방향으로 나눈다. Input도 같은 방향으로 나누고 각 GPU의 partial result를 마지막에 합친다.

연산 흐름:
1. Input을 GPU별로 나눈다.
2. 각 GPU가 partial output을 계산한다.
3. 결과를 AllReduce 또는 ReduceScatter로 합친다.

### 2.3 Column vs Row 비교

| 방식 | Weight 분할 | Input | 결과 결합 |
| --- | --- | --- | --- |
| Column Parallel | Output dimension | 동일 input 사용 | 필요하면 AllGather |
| Row Parallel | Input dimension | GPU별로 분할 | Reduce 계열 통신 |

Megatron-LM은 column parallel과 row parallel을 연속된 linear layer에 배치해 중간 communication을 줄인다. 그래도 TP는 layer마다 collective communication이 필요하므로 GPU 간 연결 속도에 민감하다.

## 3. 실제 서빙 관점

LLM inference에서는 다음 세 지표를 함께 봐야 한다.

- latency(응답 속도)
- throughput(처리량)
- GPU utilization

실제 대규모 환경에서는 한 가지 방식만 사용하기보다 PP, TP, Data Parallelism을 함께 사용한다.

```text
한 node의 빠른 GPU 연결 안에서는 TP
여러 node 사이에서는 PP
동일 model replica를 늘릴 때는 DP
```

항상 이렇게 고정되는 것은 아니지만, 통신량이 많은 TP를 NVLink 같은 빠른 연결 안에 두려는 이유를 보여준다.

대표 스택:
- Megatron-LM
- DeepSpeed
- vLLM

## 4. 내가 이해한 핵심

PP와 TP는 같은 model parallelism이지만 해결하는 문제가 다르다.

```text
Pipeline Parallelism
layer 묶음을 나눈다
-> model memory 분산
-> bubble과 stage balancing이 중요

Tensor Parallelism
layer 내부 matrix를 나눈다
-> 한 layer의 연산과 memory 분산
-> GPU 간 collective communication이 중요
```

결국 GPU가 여러 장이라는 이유만으로 병렬화 방식을 정하면 안 된다. 먼저 model이 한 GPU에 들어가는지, GPU 간 interconnect가 얼마나 빠른지, 목표가 training throughput인지 inference latency인지 확인해야 한다.

핵심은 memory, communication, latency 사이의 trade-off를 workload에 맞게 조절하는 것이다.

## 참고 자료

- <https://yjoonjang.medium.com/분산-처리-3-pipeline-parallelism과-tensor-parallelism에-관하여-7b4420fe0281>
