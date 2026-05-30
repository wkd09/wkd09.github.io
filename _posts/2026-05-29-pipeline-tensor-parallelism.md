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

이 글에서는 Model Training과 Inference에서 사용되는 다양한 Parallelism 전략을 정리한다.
단순한 개념 설명이 아니라,
각 방식이 어떤 병목을 해결하기 위해 등장했는지,
그리고 실제 서빙 환경에서 어떤 trade-off를 가지는지를 중심으로 살펴본다.
1. Pipeline Parallelism (PP)
Pipeline Parallelism은 모델을 layer 단위로 분할하여 여러 GPU에 나누는 방식이다.
단일 GPU로는 감당할 수 없는 대형 모델을 학습하거나 추론할 때 사용되며,
특히 다음과 같은 상황에서 많이 사용된다.
• 모델이 너무 커서 GPU 하나에 올라가지 않을 때
• GPU를 더 활용해서 throughput을 올리고 싶을 때
• Tensor Parallel보다 통신 비용을 줄이고 싶을 때
하지만 구조적으로 다음과 같은 trade-off가 존재한다.
Pipeline의 핵심 trade-off
1. stage 간 타이밍이 맞지 않아 bubble time 발생
2. 일부 GPU가 idle 상태가 되면서 utilization 감소
3. 이를 줄이기 위해 micro-batch를 늘리면 activation memory 증가
-> 즉, GPU 활용률을 올리려면 메모리를 더 써야 하는 구조
1.1 GPipe
이러한 문제를 해결하기 위해 등장한 것이 GPipe이다.
핵심 아이디어
• Batch를 Micro-batch로 분할
• GPU 간 pipeline을 구성하여 병렬 처리
![Pipeline/Tensor Parallelism image 1](https://blog.kakaocdn.net/dna/cV34OW/btsz1KWlruC/AAAAAAAAAAAAAAAAAAAAALu5KcbdISk9Ajyv4edwvIgqOn60cQvPZY77EM4Vzv-e/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1777561199&allow_ip=&allow_referer=&signature=kk0CzDtH2qFBtgCo3ls%2BcsQRZEk%3D)
효과
• Micro-batch 수 증가 → GPU idle 감소 (bubble 감소)
• pipeline 병렬성 증가 → throughput 향상
문제점
• Pipeline 마지막에서 flush 발생
• 해당 구간 동안 GPU가 아무 일도 하지 않음
![Pipeline/Tensor Parallelism image 2](https://blog.kakaocdn.net/dna/KPrC5/btsz6f85vnk/AAAAAAAAAAAAAAAAAAAAABlL8YoaokuSkUJh1MZUMdatQbl0oNlamr2QYwIaZ53K/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1777561199&allow_ip=&allow_referer=&signature=ZqOavXIZj3bo9%2FHIBneld%2FYZ5hg%3D)
→ 즉,
• 메모리는 효율적
• GPU utilization은 낮음
1.2 PipeDream (1F1B)
GPipe의 idle 문제를 해결하기 위해 등장한 방식
핵심 아이디어
• 1 Forward → 1 Backward 즉시 수행 (1F1B)
• GPU가 idle 상태 없이 계속 작업 수행
![Pipeline/Tensor Parallelism image 3](https://blogs.nvidia.co.kr/wp-content/uploads/sites/16/2021/05/%EC%BA%A1%EC%B2%984.jpg)
효과
• GPU utilization 증가
• pipeline flush 제거
문제점 (중요)
• Forward / Backward 사이에 파라미터가 변경됨
• → gradient inconsistency 발생
해결 방법
• 여러 버전의 parameter 저장 (weight versioning)
Trade-off
• 연산 효율 ↑
• VRAM 사용량 ↑ (매우 큼)
1.3 PP 정리
GPipe: 메모리 효율적이지만 bubble이 많다.
PipeDream: GPU 효율 높지만, VRAM 사용이 많다
2. Tensor Parallelism (TP)
Tensor Parallelism은 모델을 layer 단위가 아니라
연산 단위 (tensor 단위)로 분할하는 방식이다.
즉,
• PP: layer 기준 분할 (세로)
• TP: 연산 기준 분할 (가로)
![Pipeline/Tensor Parallelism image 4](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*pK_SrHVeKDgRvVnt.png)
대표 구현: Megatron-LM
2.1 Column Parallelism
방식
• Weight를 column 기준으로 분할
• Input은 모든 GPU에 broadcast
연산 흐름
1. 동일한 input을 모든 GPU에 전달
2. 각 GPU가 부분 계산 수행
3. 결과를 All-Gather로 합침
![Pipeline/Tensor Parallelism image 5](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*cwEIjbnxwlpLFnWU.png)
2.2 Row Parallelism
방식
• Weight를 row 기준으로 분할
• Input을 GPU별로 나눔 (scatter)
연산 흐름
1. input을 분할하여 각 GPU에 전달
2. 각 GPU에서 계산 수행
3. 결과를 element-wise sum으로 결합
![Pipeline/Tensor Parallelism image 6](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*iehOFKm67UwJRZ9n.png)
2.3 Column vs Row 비교
Column Parallelism
• 입력값은 broadcast
• 출력값은 gather
• 통신 방식은 All-Gather
• 계산 독립적
Row Parallelism
• 입력값은 scatter
• 출력값은 reduce
• 통신 방식은 All-Reduce
• 통신 효율적
4. 실제 서빙 관점에서 중요한 포인트
LLM Inference에서는 다음이 중요하다:
• latency (응답 속도)
• throughput (처리량)
• GPU utilization
→ 현실에서는 단일 방식이 아니라
Hybrid 방식 사용
• PP + TP + Data Parallelism
대표 예:
• Megatron-LM
• DeepSpeed
• vLLM
5. 결론
• GPipe → 메모리 효율
• PipeDream → GPU 효율
• Tensor Parallelism → 연산 효율
→ 결국 핵심은
메모리 vs 통신 vs latency trade-off를 어떻게 조절하느냐
## 참고 자료
- <https://yjoonjang.medium.com/분산-처리-3-pipeline-parallelism과-tensor-parallelism에-관하여-7b4420fe0281>
