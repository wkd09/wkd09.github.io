---
title: "LoRA 논문 정리: Low-Rank Adaptation으로 효율적인 파인튜닝하기"
date: 2026-05-29 01:10:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - LoRA
  - Paper
source: "Notion PDF Export - LoRA"
---

LoRA는 **Low-Rank Adaptation of Large Language Models** 논문에서 제안된 parameter-efficient fine-tuning 방법이다.

대규모 언어 모델을 full fine-tuning하면 비용이 매우 크다. GPT-3 175B 같은 모델을 downstream task마다 통째로 fine-tuning하고 저장하는 것은 비현실적이다.

LoRA는 사전학습된 weight는 고정하고, 작은 low-rank matrix만 학습해 이 문제를 줄인다.

## 문제의식

Full fine-tuning은 모델 전체 parameter를 업데이트한다.

이 방식은 성능은 좋을 수 있지만 다음 문제가 있다.

- 학습 가능한 parameter 수가 너무 많다.
- GPU 메모리 요구량이 크다.
- task마다 별도 model checkpoint를 저장해야 한다.
- 배포와 전환 비용이 크다.

기존 parameter-efficient 방법으로 adapter, prefix tuning 등이 있었지만, 각각 한계가 있다.

Adapter는 모델 중간에 layer를 추가하기 때문에 inference latency가 생길 수 있다. Prefix tuning은 input context 일부를 prefix로 사용하므로 실제 입력 길이를 희생할 수 있고, 최적화가 불안정할 수 있다.

## LoRA의 핵심 아이디어

LoRA는 weight update가 낮은 rank를 가진다고 가정한다.

기존 weight를 `W0`라고 할 때 full fine-tuning은 `W0 + ΔW`를 직접 학습한다. LoRA는 `ΔW`를 두 개의 작은 matrix 곱으로 표현한다.

```text
W = W0 + BA
```

여기서:

- `W0`: 사전학습된 weight, 고정
- `A`: 학습 가능한 low-rank matrix
- `B`: 학습 가능한 low-rank matrix
- `r`: rank, 매우 작은 값

수식으로 보면 다음과 같다.

```text
B in R^{d x r}
A in R^{r x k}
r << min(d, k)
```

즉, 큰 matrix 전체를 학습하지 않고 작은 두 matrix만 학습한다.

## Forward Pass

기존 linear layer가 다음과 같다면:

```text
h = W0 x
```

LoRA를 적용한 forward pass는 다음처럼 된다.

```text
h = W0 x + BAx
```

`W0`는 고정되어 있고, `A`, `B`만 학습된다.

초기화에서는 보통 `A`를 random Gaussian으로 초기화하고 `B`를 0으로 초기화한다. 그러면 학습 시작 시점에는 `BA = 0`이므로 원래 모델 동작을 해치지 않는다.

## LoRA의 장점

LoRA의 장점은 명확하다.

- 학습 parameter 수가 크게 줄어든다.
- GPU 메모리 요구량이 줄어든다.
- task별 adapter를 작게 저장할 수 있다.
- 배포 시 `W0 + BA`를 merge하면 추가 inference latency가 없다.
- task 전환 시 LoRA weight만 교체하면 된다.

논문에서는 GPT-3 175B 기준으로 학습 가능한 parameter 수를 크게 줄이고, GPU 메모리 요구량도 낮출 수 있음을 보인다.

## Adapter, Prefix Tuning과의 차이

Adapter는 activation space에 별도 module을 추가한다. 그래서 구조적으로 추가 연산이 생긴다.

Prefix tuning은 input space를 확장한다. 하지만 context length 일부를 prefix가 차지한다.

LoRA는 parameter space에서 update 방향을 low-rank subspace로 제한한다. 그래서 모델 구조를 크게 바꾸지 않고, inference 시 merge를 통해 latency 증가를 피할 수 있다.

## 한계

LoRA에도 한계는 있다.

- 어떤 weight matrix에 LoRA를 적용할지 선택해야 한다.
- rank `r`을 어떻게 잡을지 실험이 필요하다.
- 하나의 batch 안에서 서로 다른 task의 LoRA module을 섞어 처리하기는 복잡할 수 있다.

그래도 실용적으로는 LLM fine-tuning 비용을 크게 낮추는 강력한 방법이다.

## 정리

LoRA는 full fine-tuning의 비용 문제를 low-rank update로 해결한다.

핵심은 다음과 같다.

- 사전학습 weight는 고정한다.
- `ΔW`를 `BA`로 분해해 작은 matrix만 학습한다.
- 학습 parameter와 GPU 메모리를 줄인다.
- merge하면 inference latency가 늘지 않는다.

LLM을 특정 domain이나 task에 맞추고 싶지만 full fine-tuning 비용이 부담될 때 가장 먼저 고려할 만한 방법이다.

## 참고

- Paper: <https://arxiv.org/abs/2106.09685>
- GitHub: <https://github.com/microsoft/LoRA>
