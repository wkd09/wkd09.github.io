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

이 글에서는 모델 학습/추론에서 자주 쓰는 병렬화 전략인 Pipeline Parallelism(PP)과 Tensor Parallelism(TP)을 정리한다.

## 1. Pipeline Parallelism (PP)

Pipeline Parallelism은 모델을 레이어 단위로 나눠 여러 GPU에 배치하는 방식이다.

주요 사용 상황:
- 모델이 커서 단일 GPU 메모리에 올라가지 않을 때
- GPU 활용률과 처리량(throughput)을 높이고 싶을 때
- Tensor Parallelism 대비 통신 패턴을 단순화하고 싶을 때

핵심 trade-off:
1. 스테이지 간 타이밍 불일치로 bubble time이 생긴다.
2. 일부 GPU가 idle 상태가 되어 utilization이 떨어진다.
3. bubble을 줄이려 micro-batch를 늘리면 activation memory가 증가한다.

요약하면, GPU 활용률을 높이기 위해 메모리를 더 쓰는 구조다.

### 1.1 GPipe

핵심 아이디어:
- batch를 micro-batch로 쪼갠다.
- GPU 스테이지를 파이프라인으로 연결해 병렬 처리한다.

장점:
- micro-batch 수가 늘어 GPU idle이 감소한다.
- pipeline 병렬성이 올라 throughput이 향상된다.

단점:
- 파이프라인 마지막 flush 구간에서 idle이 발생한다.

정리:
- 메모리는 상대적으로 효율적
- GPU utilization은 제한될 수 있음

### 1.2 PipeDream (1F1B)

GPipe의 idle 문제를 줄이기 위한 방식이다.

핵심 아이디어:
- `1 Forward -> 1 Backward`를 즉시 수행(1F1B)
- 각 GPU가 쉬지 않고 연산을 이어간다.

장점:
- GPU utilization 증가
- pipeline flush 비용 감소

주의점:
- Forward와 Backward 사이에 파라미터 버전이 달라져 gradient inconsistency가 생길 수 있다.
- 이를 막기 위해 weight versioning이 필요하다.

trade-off:
- 연산 효율 상승
- VRAM 사용량 증가

### 1.3 PP 요약

- GPipe: 메모리 효율은 좋지만 bubble이 상대적으로 큼
- PipeDream: GPU 효율은 좋지만 VRAM 부담이 큼

## 2. Tensor Parallelism (TP)

Tensor Parallelism은 레이어를 통째로 나누는 대신, 레이어 내부 연산(텐서)을 GPU들로 분할한다.

- PP: 레이어 기준 분할(세로)
- TP: 연산 기준 분할(가로)

대표 구현: Megatron-LM

### 2.1 Column Parallelism

방식:
- weight를 column 기준으로 분할
- input은 모든 GPU에 broadcast

연산 흐름:
1. 동일 input을 각 GPU에 전달
2. 각 GPU가 부분 계산
3. 결과를 All-Gather로 결합

### 2.2 Row Parallelism

방식:
- weight를 row 기준으로 분할
- input을 GPU별로 scatter

연산 흐름:
1. input 분할 전달
2. 각 GPU에서 부분 계산
3. 결과를 reduce(보통 All-Reduce/합산)로 결합

### 2.3 Column vs Row 비교

Column Parallelism:
- 입력 broadcast
- 출력 gather
- 통신 패턴은 All-Gather 중심

Row Parallelism:
- 입력 scatter
- 출력 reduce
- 통신 패턴은 All-Reduce 중심

## 3. 실제 서빙 관점

LLM inference에서는 아래 3개를 함께 본다.
- latency(응답 속도)
- throughput(처리량)
- GPU utilization

실무에서는 단일 전략보다 Hybrid 구성이 일반적이다.
- PP + TP + Data Parallelism

대표 스택:
- Megatron-LM
- DeepSpeed
- vLLM

## 4. 결론

- GPipe: 메모리 효율 중심
- PipeDream: GPU 효율 중심
- Tensor Parallelism: 연산 분산 효율 중심

핵심은 메모리, 통신, latency 사이 trade-off를 워크로드에 맞게 조절하는 것이다.
## 참고 자료
- <https://yjoonjang.medium.com/분산-처리-3-pipeline-parallelism과-tensor-parallelism에-관하여-7b4420fe0281>
