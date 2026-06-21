---
title: "Long Context에서 KV Cache가 병목이 되는 이유"
date: 2026-06-21 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - LLM
  - Serving
  - KV Cache
  - Long Context
source: "Medium - Large KV Cache, Hugging Face Blog - KV Caching"
---

LLM을 서빙할 때 `context window`를 늘리면 모델이 더 많은 정보를 참고할 수 있다.

하지만 long context는 공짜가 아니다. 입력 토큰이 길어질수록 attention 계산량도 늘어나고, 특히 autoregressive generation에서는 **KV cache 메모리**가 빠르게 커진다. 모델 weight는 한 번 GPU에 올라가면 거의 고정 비용에 가깝지만, KV cache는 요청 수, context length, 출력 길이에 따라 계속 증가하는 동적 비용이다.

이 글에서는 KV cache가 왜 필요한지, 얼마나 많은 메모리를 쓰는지, 그리고 long context serving에서 왜 병목이 되는지 정리한다.

## KV Cache가 필요한 이유

GPT 계열 모델은 autoregressive 방식으로 토큰을 하나씩 생성한다.

예를 들어 모델이 다음 문장을 생성한다고 해보자.

```text
The weather today is
```

다음 토큰을 만들 때 모델은 앞의 모든 토큰을 참고한다. 그리고 새 토큰을 하나 더 생성하면, 다시 전체 prefix를 참고해 다음 토큰을 만든다.

KV cache가 없다면 매 generation step마다 이전 토큰들의 `Key`, `Value`를 다시 계산해야 한다. 이미 계산했던 prefix에 대해 같은 연산을 반복하는 셈이다.

KV cache는 이 반복 계산을 줄이기 위해 attention layer에서 만들어진 `Key`, `Value` tensor를 저장해둔다. 다음 토큰을 생성할 때는 새 토큰의 `Query`만 계산하고, 이전 토큰들의 `Key`, `Value`는 cache에서 가져와 재사용한다.

즉, KV cache는 다음 trade-off다.

| 항목 | KV cache 없음 | KV cache 사용 |
| --- | --- | --- |
| 이전 토큰 K/V | 매 step 다시 계산 | cache에서 재사용 |
| 연산량 | 큼 | 줄어듦 |
| 메모리 사용량 | 상대적으로 작음 | cache 저장 공간 필요 |
| 긴 생성 속도 | 느려지기 쉬움 | 훨씬 유리함 |

Hugging Face 글의 예시처럼, 실제 generation loop에서는 매 step마다 새 token의 `K`, `V`를 cache 뒤에 append하고, 다음 step에서 이 누적 cache를 다시 참조한다.

```text
step 1: cache = [K1, V1]
step 2: cache = [K1, K2], [V1, V2]
step 3: cache = [K1, K2, K3], [V1, V2, V3]
...
```

이 구조 덕분에 decoding 단계에서는 이전 token들의 K/V projection을 반복하지 않는다. 대신 cache는 계속 길어진다.

## Prefill과 Decode

LLM 추론은 크게 두 단계로 나눠볼 수 있다.

첫 번째는 **prefill**이다. 사용자가 입력한 prompt 전체를 한 번에 처리하면서 각 layer의 KV cache를 만든다. 긴 문서를 prompt로 넣으면 이 단계가 무거워진다.

두 번째는 **decode**다. 모델이 새 토큰을 하나씩 생성하는 단계다. 각 step에서는 새 토큰의 `Q`, `K`, `V`를 계산하고, 이전 token들의 `K`, `V`는 cache에서 가져온다.

KV cache는 특히 decode에서 중요하다. 답변이 길어질수록 이전 token들을 계속 다시 계산하지 않아도 되기 때문이다.

다만 prefill에서 만들어진 cache와 decode 중 새로 추가되는 cache는 모두 GPU memory를 차지한다. 그래서 긴 prompt와 긴 output이 동시에 오면 cache가 빠르게 커진다.

## KV Cache 메모리 계산식

KV cache의 크기는 대략 다음 요소로 결정된다.

- batch size
- sequence length
- layer 수
- KV head 수
- head dimension
- data type 크기

계산식은 다음처럼 쓸 수 있다.

$$
KV\ Cache\ Memory
= 2 \times B \times S \times L \times H_{kv} \times D \times bytes
$$

각 기호의 의미는 다음과 같다.

| 기호 | 의미 |
| --- | --- |
| `2` | Key와 Value 두 tensor |
| `B` | batch size |
| `S` | sequence length |
| `L` | transformer layer 수 |
| `H_{kv}` | KV head 수 |
| `D` | head dimension |
| `bytes` | FP16이면 2 bytes, FP32이면 4 bytes |

MHA에서는 `H_{kv}`가 attention head 수와 같다. 반면 GQA나 MQA에서는 여러 query head가 더 적은 수의 KV head를 공유한다. 그래서 같은 모델 크기라도 GQA/MQA를 쓰면 KV cache 메모리를 크게 줄일 수 있다.

중요한 점은 KV cache가 **batch size와 sequence length에 선형적으로 증가**한다는 것이다.

```text
batch size 2배  -> KV cache 2배
sequence length 2배 -> KV cache 2배
```

하나만 보면 선형 증가라 괜찮아 보일 수 있다. 하지만 serving에서는 batch size와 sequence length가 같이 커진다. 긴 prompt를 가진 요청을 여러 개 동시에 처리하면 KV cache가 순식간에 GPU memory를 잡아먹는다.

## 간단한 예시

LLaMA 계열 13B급 모델을 단순화해서 보자.

```text
layers = 40
hidden size = 5120
MHA 기준 KV heads * head dim = 5120
dtype = FP16 = 2 bytes
```

MHA 기준 token 하나가 모든 layer에 남기는 KV cache는 다음과 비슷하다.

$$
2 \times 40 \times 5120 \times 2
= 819,200\ bytes
\approx 0.78\ MiB
$$

즉, batch 안의 token 하나가 대략 0.8MiB 정도의 KV cache를 만든다.

sequence length가 4K라면 batch 1개만으로도 대략 3GiB 이상이 필요하다.

```text
0.78 MiB * 4096 tokens ~= 3.2 GiB
```

sequence length를 32K로 늘리면 같은 batch 1에서도 25GiB 이상이 된다.

```text
0.78 MiB * 32768 tokens ~= 25.6 GiB
```

여기에 batch size가 4라면 약 100GiB 수준이 된다. 모델 weight까지 GPU에 올라가야 하므로, long context serving에서 KV cache가 얼마나 빠르게 병목이 되는지 감이 온다.

## Long Context에서 왜 더 문제가 되는가

LLM의 context window가 커지면 사용자는 더 긴 문서, 더 긴 대화, 더 많은 검색 결과를 한 번에 넣고 싶어 한다.

하지만 serving 입장에서 long context는 다음 문제를 만든다.

첫째, **동시 처리 가능한 요청 수가 줄어든다.** GPU memory가 한정되어 있기 때문에 요청 하나가 긴 KV cache를 차지하면 batch size를 키우기 어렵다.

둘째, **처리량이 떨어진다.** 같은 GPU에서도 짧은 요청 여러 개를 처리할 수 있던 자리에 긴 요청 하나가 들어오면 전체 tokens/sec가 낮아질 수 있다.

셋째, **메모리 fragmentation과 scheduling 문제가 커진다.** 요청마다 prompt 길이와 output 길이가 다르기 때문에 KV cache를 고정 크기로 잡으면 낭비가 생기고, 동적으로 잡으면 관리가 어려워진다.

넷째, **운영 설정이 까다로워진다.** `max_model_len`, 동시 요청 수, batch 정책, cache block 크기, GPU memory utilization 같은 값을 잘못 잡으면 OOM이 나거나 처리량이 급격히 떨어질 수 있다.

결국 long context는 모델의 기능을 늘려주지만, 서빙 시스템에는 memory capacity 문제를 가져온다.

## Weight보다 KV Cache가 더 커질 수 있다

모델 weight는 모델을 로드할 때 필요한 고정 비용이다.

예를 들어 FP16 기준 70B 모델 weight는 대략 140GB 수준이다. tensor parallelism으로 여러 GPU에 나누어 올릴 수 있지만, 전체적으로 큰 고정 비용인 것은 맞다.

반면 KV cache는 요청 상태에 따라 달라진다.

```text
짧은 prompt + 낮은 동시성 -> KV cache 작음
긴 prompt + 높은 동시성 -> KV cache 큼
긴 prompt + 긴 output + 높은 동시성 -> KV cache 매우 큼
```

그래서 long context 환경에서는 어느 순간 weight보다 KV cache가 더 중요한 병목이 된다. 모델을 올리는 데는 성공했는데 실제 traffic을 받으면 OOM이 나는 이유가 여기에 있다.

## KV Cache를 줄이는 방법

KV cache 병목을 줄이는 방법은 여러 층위가 있다.

### 1. GQA와 MQA

MHA에서는 query head마다 key/value head가 대응된다. GQA는 여러 query head가 일부 KV head를 공유하고, MQA는 모든 query head가 하나의 KV head를 공유하는 식으로 cache를 줄인다.

KV cache 계산식에서 `H_{kv}`가 줄어드는 것이므로 효과가 직접적이다.

예를 들어 attention head가 32개인 모델에서 KV head를 8개만 쓰면, 단순 계산으로 KV cache의 head 차원 비용이 1/4 수준으로 줄어든다.

### 2. KV Cache Quantization

모델 weight만 양자화할 수 있는 것이 아니라 KV cache도 낮은 precision으로 저장할 수 있다.

FP16 대신 INT8이나 더 낮은 bit-width로 cache를 저장하면 memory footprint를 줄일 수 있다. 다만 attention score와 generation 품질에 영향을 줄 수 있으므로 모델과 workload별 검증이 필요하다.

### 3. PagedAttention

vLLM의 PagedAttention은 KV cache를 연속된 큰 memory block으로 관리하지 않고, 작은 block 단위로 나누어 관리한다.

운영체제의 virtual memory처럼 요청별 block table을 두고 필요한 cache block만 연결해 쓰는 방식이다. 이러면 요청마다 길이가 달라도 낭비되는 reserved memory와 fragmentation을 줄일 수 있다.

특히 serving에서는 모든 요청의 길이가 같지 않다. PagedAttention은 이런 variable-length workload에서 KV cache memory utilization을 높이는 데 유리하다.

### 4. Context Length 제한과 정책

가장 현실적인 방법은 context length를 무조건 크게 열어두지 않는 것이다.

서비스에서는 다음 정책이 필요할 수 있다.

- 사용자 tier별 max context length 제한
- retrieval 결과 개수 제한
- 오래된 대화 history 요약
- prompt 압축
- output token limit 조정
- 긴 요청과 짧은 요청의 batch 분리

모델이 128K context를 지원한다고 해서 모든 요청에 128K를 허용해야 하는 것은 아니다. serving 비용과 latency 목표에 맞춰 실제 운영 context를 정해야 한다.

## 개발자가 확인해야 할 지표

KV cache 병목을 보려면 단순히 GPU 사용률만 보면 부족하다.

다음 지표를 함께 확인하는 것이 좋다.

- GPU memory 중 model weight와 KV cache가 차지하는 비율
- prefill latency
- decode tokens/sec
- time to first token
- request당 prompt length와 output length 분포
- batch size와 동시 요청 수
- OOM 발생 시점의 sequence length
- cache hit, eviction, block utilization

특히 평균 prompt length보다 p95, p99 prompt length가 중요하다. 일부 긴 요청이 전체 scheduler와 memory를 흔들 수 있기 때문이다.

## 정리

KV cache는 LLM inference를 빠르게 만드는 핵심 최적화다.

이전 token들의 `Key`, `Value`를 재사용하기 때문에 decode 단계의 반복 계산을 크게 줄일 수 있다. 하지만 그 대가로 GPU memory를 사용한다.

long context에서는 이 trade-off가 더 날카로워진다. sequence length가 길어지고 batch size가 커질수록 KV cache는 선형적으로 증가한다. 운영 환경에서는 이 선형 증가가 곧 memory capacity, throughput, latency 문제로 이어진다.

따라서 LLM serving을 설계할 때는 모델 weight 크기만 볼 것이 아니라 KV cache까지 함께 계산해야 한다.

핵심은 다음과 같다.

- KV cache는 연산량을 줄이는 대신 메모리를 사용한다.
- KV cache 크기는 `batch size * sequence length * layer 수 * KV head 수 * head dimension`에 비례한다.
- long context와 높은 동시성에서는 KV cache가 weight보다 더 큰 병목이 될 수 있다.
- GQA/MQA, cache quantization, PagedAttention, context 정책이 주요 대응 방법이다.
- 실제 서비스에서는 평균보다 p95/p99 길이와 동시 요청 분포를 봐야 한다.

결국 long context serving의 핵심 질문은 "모델이 몇 token까지 받을 수 있는가?"가 아니다.

"우리가 가진 GPU memory 안에서, 그 context length를 몇 명에게 동시에 안정적으로 제공할 수 있는가?"가 더 실전적인 질문이다.

## 참고 자료

- [Long Context로 인한 Large KV Cache의 문제점과 해결 방안: Part I-KV cache의 메모리 요구량](https://moon-walker.medium.com/long-context%EB%A1%9C-%EC%9D%B8%ED%95%9C-large-kv-cache%EC%9D%98-%EB%AC%B8%EC%A0%9C%EC%A0%90%EA%B3%BC-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EC%95%88-part-i-kv-cache%EC%9D%98-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EC%9A%94%EA%B5%AC%EB%9F%89-025f3d5dea93)
- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
