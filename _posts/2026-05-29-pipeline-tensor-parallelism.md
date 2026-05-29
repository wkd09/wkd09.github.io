---
title: "Pipeline Parallelism과 Tensor Parallelism 정리"
date: 2026-05-29 00:20:00 +0900
categories:
  - engineering
tags:
  - AI
  - Training
  - Inference
  - Parallelism
source: "Velog PDF - Pipeline/Tensor Parallelism"
---

대형 모델을 학습하거나 추론할 때는 단일 GPU에 모델 전체를 올리기 어렵다. 이때 모델을 어떻게 나눌 것인지가 중요해진다.

대표적인 방식은 Pipeline Parallelism과 Tensor Parallelism이다.

- Pipeline Parallelism: layer 단위로 모델을 나눈다.
- Tensor Parallelism: 연산 또는 tensor 단위로 모델을 나눈다.

두 방식 모두 GPU 메모리 한계를 넘기 위해 사용되지만, 병목과 trade-off가 다르다.

## Pipeline Parallelism

Pipeline Parallelism은 모델을 layer 단위로 잘라 여러 GPU에 나누는 방식이다.

예를 들어 24개 layer가 있는 모델을 4개 GPU에 나눈다면, 각 GPU가 6개 layer를 담당한다. 입력은 첫 번째 GPU에서 시작해 다음 GPU로 순서대로 전달된다.

주로 다음 상황에서 사용한다.

- 모델이 너무 커서 GPU 하나에 올라가지 않을 때
- GPU를 더 활용해 throughput을 올리고 싶을 때
- Tensor Parallel보다 통신 비용을 줄이고 싶을 때

하지만 pipeline 구조에는 bubble time이 생긴다. 앞 stage가 계산을 끝내고 다음 stage가 받을 때까지 일부 GPU가 idle 상태가 될 수 있다.

## GPipe

GPipe는 batch를 micro-batch로 나누어 pipeline 병렬성을 높이는 방식이다.

핵심 아이디어는 다음과 같다.

- 큰 batch를 여러 micro-batch로 분할한다.
- 각 micro-batch를 pipeline stage에 순차적으로 흘린다.
- stage들이 동시에 다른 micro-batch를 처리하게 만든다.

micro-batch 수가 늘어나면 GPU idle이 줄고 throughput이 올라간다.

하지만 마지막에는 pipeline flush가 필요하다. forward와 backward가 모두 끝날 때까지 일부 stage가 기다리는 구간이 생긴다.

정리하면 GPipe는 메모리 효율은 좋지만 bubble이 남는다.

## PipeDream과 1F1B

PipeDream은 GPipe의 idle 문제를 줄이기 위해 등장했다.

핵심은 1F1B, 즉 forward 하나를 수행한 뒤 backward 하나를 바로 수행하는 방식이다. 이렇게 하면 GPU가 쉬는 시간을 줄일 수 있다.

장점은 GPU utilization이 올라간다는 점이다.

하지만 문제가 있다. forward와 backward 사이에 parameter가 변경될 수 있어 gradient inconsistency가 생긴다. 이를 해결하려면 여러 버전의 parameter를 저장하는 weight versioning이 필요하다.

즉, PipeDream은 연산 효율은 높지만 VRAM 사용량이 커진다.

## Tensor Parallelism

Tensor Parallelism은 layer를 통째로 나누지 않고, layer 안의 행렬 연산을 여러 GPU에 나누어 계산한다.

대표 구현으로 Megatron-LM이 있다.

Tensor Parallelism은 특히 Transformer의 큰 matrix multiplication을 나눌 때 사용된다.

## Column Parallelism

Column Parallelism은 weight matrix를 column 기준으로 나누는 방식이다.

흐름은 다음과 같다.

1. 동일한 input을 모든 GPU에 broadcast한다.
2. 각 GPU가 자기 column shard에 대한 부분 계산을 수행한다.
3. 결과를 all-gather로 합친다.

계산은 비교적 독립적이지만, 결과를 모으는 gather 통신이 필요하다.

## Row Parallelism

Row Parallelism은 weight matrix를 row 기준으로 나누는 방식이다.

흐름은 다음과 같다.

1. input을 GPU별로 나눈다.
2. 각 GPU가 자기 shard로 계산한다.
3. 결과를 element-wise sum 또는 all-reduce로 결합한다.

Row Parallelism은 출력 결합을 위해 reduce 통신이 필요하다.

## Column vs Row

Column Parallelism:

- input은 broadcast
- output은 gather
- 통신 방식은 all-gather
- 부분 계산은 독립적

Row Parallelism:

- input은 scatter
- output은 reduce
- 통신 방식은 all-reduce
- 통신 효율을 잘 설계해야 함

## 실제 서빙 관점

LLM inference에서는 latency, throughput, GPU utilization이 모두 중요하다.

현실에서는 단일 병렬화 방식만 쓰지 않는다. 보통 다음을 조합한다.

- Pipeline Parallelism
- Tensor Parallelism
- Data Parallelism

예를 들어 모델이 GPU 하나에 올라가지 않으면 PP 또는 TP를 사용하고, 요청 처리량을 늘리기 위해 DP를 함께 사용한다.

## 정리

- GPipe는 micro-batch로 pipeline bubble을 줄이지만 flush 구간이 남는다.
- PipeDream은 1F1B로 GPU utilization을 높이지만 weight versioning 때문에 VRAM을 더 쓴다.
- Tensor Parallelism은 layer 내부 연산을 나누어 대형 matrix 연산을 여러 GPU에서 처리한다.
- 실제 시스템은 PP, TP, DP를 함께 사용한다.

결국 핵심은 메모리, 통신 비용, latency 사이의 trade-off를 어떻게 조절하느냐다.
