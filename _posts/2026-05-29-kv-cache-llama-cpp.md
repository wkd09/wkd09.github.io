---
title: "LLM 서빙 기초: KV Cache와 llama.cpp"
date: 2026-05-29 01:00:00 +0900
last_modified_at: 2026-07-04 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - LLM
  - Serving
  - KV Cache
source: "Notion PDF Export - KV Cache, llama.cpp"
---

LLM을 실제로 서빙할 때는 모델 정확도만큼 추론 속도와 메모리 사용량이 중요하다.

이 글에서는 LLM inference에서 자주 등장하는 KV Cache와 로컬 추론 엔진인 llama.cpp를 정리한다.

## KV Cache가 필요한 이유

LLM은 autoregressive 방식으로 다음 토큰을 하나씩 생성한다.

예를 들어 이미 `Hello, how are`까지 생성했고 다음 토큰을 만들 때, 모델은 이전 토큰들의 key와 value를 다시 계산할 필요가 있다. 아무 최적화가 없다면 매번 전체 prefix에 대해 같은 계산을 반복하게 된다.

KV Cache는 이전 토큰의 Key, Value를 저장해두고 다음 토큰 생성 시 재사용하는 방식이다.

## KV Cache 동작 과정

흐름은 다음과 같다.

1. 입력 토큰의 Key와 Value를 계산한다.
2. 계산된 Key와 Value를 cache에 저장한다.
3. 다음 토큰을 생성할 때 새 토큰의 Query를 계산한다.
4. 이전 Key, Value는 cache에서 가져온다.
5. 새 token의 Key, Value를 cache에 추가한다.
6. 이 과정을 반복한다.

즉, 매 step마다 전체 sequence의 Key, Value를 다시 계산하지 않고, 새 토큰에 필요한 부분만 추가한다.

## 일반 추론과 KV Cache 비교

일반 추론은 매번 이전 token들에 대한 계산을 반복한다. sequence가 길어질수록 계산량이 계속 증가한다.

KV Cache는 이전 계산 결과를 저장하기 때문에 추가 메모리를 사용한다. 대신 긴 context에서도 반복 계산을 줄여 속도를 크게 개선한다.

정리하면 다음과 같다.

| 항목 | 일반 추론 | KV Cache |
| --- | --- | --- |
| 계산 방식 | 이전 token을 반복 계산 | 이전 K/V 재사용 |
| 메모리 | 상대적으로 적게 사용 | cache 저장 메모리 필요 |
| 속도 | sequence가 길수록 느려짐 | 긴 sequence에서 효율적 |
| 병목 | 반복 계산 | KV cache 메모리 |

Notion 실험 기록에서는 `use_cache=True`일 때 약 8.2초, `use_cache=False`일 때 약 40.4초로 약 4.9배 차이가 났다.

## KV Cache의 trade-off

KV Cache는 속도를 높이지만 공짜는 아니다.

동시 요청이 많거나 context length가 길어지면 KV cache가 GPU 메모리를 크게 차지한다. 그래서 vLLM의 PagedAttention처럼 KV cache를 효율적으로 관리하는 기법이 중요해진다.

즉, KV Cache는 연산량을 줄이는 대신 메모리를 더 쓰는 최적화다.

## llama.cpp

llama.cpp는 C/C++ 기반 LLM 추론 엔진이다.

특징은 다음과 같다.

- 로컬 환경에서 모델을 실행하기 쉽다.
- CPU만으로도 실행 가능하다.
- Mac 환경에서도 비교적 잘 동작한다.
- GGUF 포맷과 양자화를 지원한다.

## GGUF

llama.cpp는 GGUF라는 모델 포맷을 사용한다.

GGUF는 모델을 빠르게 로딩하고, 양자화된 weight를 담을 수 있는 포맷이다. Hugging Face 모델을 llama.cpp에서 사용하려면 GGUF로 변환하는 과정이 필요하다.

예시는 다음과 같다.

```bash
python llama.cpp/convert-hf-to-gguf.py ./Orion-14B-Chat \
  --outfile ./Orion-14B-Chat.gguf
```

## 양자화

GGUF로 변환한 뒤에는 quantize를 통해 모델 크기를 줄일 수 있다.

```bash
./llama.cpp/quantize \
  ./Orion-14B-Chat.gguf \
  ./orion-14b-chat.Q5_K_S.gguf \
  q5_k_s
```

llama.cpp에는 여러 양자화 옵션이 있다.

- `Q2_K`
- `Q3_K_S`
- `Q3_K_M`
- `Q3_K_L`
- `Q4_K_S`
- `Q4_K_M`
- `Q5_K_S`
- `Q5_K_M`
- `Q6_K`

bit 수가 낮아질수록 모델 크기와 메모리 사용량은 줄어들지만, 품질 손실 가능성이 커진다.

## 정리

LLM 서빙에서는 다음 두 가지를 같이 봐야 한다.

- KV Cache: 반복 계산을 줄여 generation 속도를 높인다.
- llama.cpp: 로컬/저사양 환경에서 LLM을 실행하기 쉽게 해준다.

KV Cache는 추론 속도 최적화의 기본이고, llama.cpp와 GGUF/quantization은 모델을 제한된 환경에서 실행하기 위한 실용적인 도구다.
