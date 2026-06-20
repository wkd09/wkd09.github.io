---
title: "FlashAttention 정리: Attention 병목을 IO 관점에서 줄이는 방법"
date: 2026-06-16 00:00:00 +0900
categories:
  - engineering
tags:
  - AI
  - LLM
  - Transformer
  - Attention
  - GPU
source: "Draft - FlashAttention"
---

Transformer에서 attention은 가장 중요한 연산 중 하나지만, sequence length가 길어질수록 비용이 빠르게 커진다.

특히 LLM에서 context length를 늘리거나 batch size를 키우면 attention이 학습과 추론의 병목이 되기 쉽다. FlashAttention은 이 병목을 단순히 연산량 관점이 아니라 **GPU memory IO 관점**에서 해결하려는 방법이다.

이 글에서는 FlashAttention이 왜 필요한지, 어떤 방식으로 memory access를 줄이는지, 그리고 일반 attention과 어떤 차이가 있는지 정리한다.

## 기존 Attention의 문제

Scaled dot-product attention은 보통 다음과 같이 계산한다.

$$
Attention(Q, K, V)
= softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

여기서 `Q`, `K`, `V`는 각각 query, key, value이다.

계산 흐름은 단순하게 보면 다음과 같다.

1. `QK^T`를 계산해 attention score matrix를 만든다.
2. score에 softmax를 적용한다.
3. softmax 결과에 `V`를 곱해 output을 만든다.

문제는 `QK^T` 결과가 sequence length 기준으로 `N x N` 크기라는 점이다.

예를 들어 sequence length가 `4096`이면 attention score matrix는 `4096 x 4096` 크기가 된다. head와 batch까지 고려하면 중간 결과가 매우 커진다.

즉, attention은 단순히 계산량만 큰 것이 아니라 중간 matrix를 GPU memory에 쓰고 다시 읽는 비용도 크다.

![GPT-2 attention에서 PyTorch와 FlashAttention fused kernel 비교](/assets/images/blog/flashattention-gpt2-fused-kernel.png)

*FlashAttention은 attention 내부의 matmul, mask, softmax, dropout 같은 단계를 fused kernel로 처리해 중간 결과의 memory read/write를 줄인다.*

## GPU Memory Hierarchy

FlashAttention을 이해하려면 GPU memory 구조를 먼저 봐야 한다.

GPU에는 여러 종류의 memory가 있다.

- HBM: 용량은 크지만 상대적으로 느린 global memory
- SRAM/shared memory: 용량은 작지만 매우 빠른 on-chip memory
- register: 각 thread가 사용하는 가장 빠른 작은 저장 공간

![FlashAttention 원문 Figure: runtime과 sparsity speedup](/assets/images/blog/flashattention-paper-figure.png)

*원 논문 Figure 2는 standard attention, FlashAttention, block-sparse FlashAttention의 runtime 차이를 비교한다.*

![FlashAttention의 memory hierarchy와 IO-aware attention 구조](/assets/images/blog/flashattn.png)

*FlashAttention은 HBM에 큰 attention matrix를 저장하지 않고, SRAM에서 block 단위로 계산해 HBM read/write를 줄인다.*

일반적인 attention 구현은 큰 중간 matrix를 HBM에 저장했다가 다시 읽는다. 이때 실제 병목은 `QK^T`를 계산하는 FLOPs보다 HBM read/write에서 생길 수 있다.

FlashAttention의 핵심은 이 부분이다.

> attention을 더 빠르게 하려면 계산량만 볼 것이 아니라, GPU memory 사이에서 데이터를 얼마나 읽고 쓰는지도 봐야 한다.

이런 관점을 **IO-aware**하다고 표현한다.

## FlashAttention의 핵심 아이디어

FlashAttention은 attention score matrix 전체를 한 번에 만들지 않는다.

대신 `Q`, `K`, `V`를 block 단위로 나누고, 각 block을 빠른 SRAM에 올려서 계산한다. 그리고 필요한 output만 누적한 뒤 HBM에 저장한다.

핵심은 다음 두 가지다.

1. `N x N` attention matrix를 HBM에 저장하지 않는다.
2. softmax를 block 단위로 계산하면서도 전체 softmax와 같은 결과를 유지한다.

즉, FlashAttention은 approximate attention이 아니다. attention 결과를 근사하는 것이 아니라, 같은 결과를 더 메모리 효율적으로 계산하는 방식이다.

## Tiling

FlashAttention은 tiling을 사용한다.

Tiling은 큰 matrix 연산을 작은 block 연산으로 나누는 방법이다.

예를 들어 전체 `Q`, `K`, `V`를 한 번에 처리하지 않고 다음처럼 나눈다.

- `Q` block 하나를 SRAM에 올린다.
- `K`, `V` block을 순서대로 SRAM에 올린다.
- 해당 block 조합에 대한 attention을 계산한다.
- output을 누적한다.
- 다음 block으로 넘어간다.

이 방식은 GPU의 빠른 on-chip memory를 더 잘 활용한다.

중요한 점은 `QK^T` 전체 matrix를 만들지 않는다는 것이다. 필요한 block의 score만 계산하고, softmax와 value 곱까지 이어서 처리한 뒤 버린다.

## Online Softmax

FlashAttention이 어려운 이유는 softmax 때문이다.

softmax는 전체 score를 보고 normalization해야 한다.

$$
softmax(x_i)
= \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

그런데 block 단위로 attention score를 계산하면 전체 score를 한 번에 볼 수 없다. 단순히 block마다 softmax를 따로 계산하면 전체 softmax와 다른 결과가 나온다.

FlashAttention은 이를 해결하기 위해 online softmax를 사용한다.

각 block을 처리하면서 다음 값을 함께 관리한다.

- 지금까지 본 score의 최대값
- softmax denominator의 누적값
- output의 누적값

![FlashAttention의 block-wise online softmax 계산 흐름](/assets/images/blog/flash%20attn.webp)

*각 score block은 SRAM에서 계산되고, denominator와 output을 누적 보정하면서 전체 softmax와 같은 결과를 유지한다.*

새 block이 들어오면 기존 누적값을 새 최대값 기준으로 보정한 뒤 이어서 계산한다. 이렇게 하면 전체 score matrix를 저장하지 않아도, 모든 score를 본 것과 같은 softmax 결과를 만들 수 있다.

정리하면 FlashAttention은 block 단위로 계산하지만, softmax는 전체 sequence 기준으로 정확하게 유지한다.

## 일반 Attention과 FlashAttention 비교

일반 attention은 중간 attention matrix를 만든다.

```text
Q, K, V
  -> QK^T
  -> softmax(QK^T)
  -> softmax(QK^T) V
  -> output
```

FlashAttention은 중간 matrix를 HBM에 저장하지 않고 block 단위로 바로 output을 누적한다.

```text
Q block, K block, V block
  -> block score
  -> online softmax update
  -> output accumulate
  -> final output
```

차이를 표로 정리하면 다음과 같다.

| 항목 | 일반 Attention | FlashAttention |
| --- | --- | --- |
| 중간 attention matrix | HBM에 저장 | 저장하지 않음 |
| memory access | 큼 | 줄어듦 |
| 결과 | 정확한 attention | 정확한 attention |
| 핵심 최적화 | matmul 중심 | IO-aware tiling |
| 긴 sequence | memory 병목이 큼 | 더 효율적 |

## 왜 긴 Context에서 중요한가

Attention score matrix는 sequence length에 대해 quadratic하게 커진다.

sequence length를 2배로 늘리면 attention score matrix 크기는 4배가 된다. 그래서 long context를 다루는 모델에서는 attention memory 비용이 빠르게 문제가 된다.

FlashAttention은 attention score matrix를 materialize하지 않기 때문에 memory 사용량을 크게 줄일 수 있다. 이 덕분에 더 긴 sequence를 학습하거나 추론할 때 유리하다.

다만 FlashAttention이 attention의 이론적 계산 복잡도 자체를 `O(N^2)`에서 `O(N)`으로 바꾸는 것은 아니다.

여전히 모든 query-key 조합을 계산해야 한다. 대신 중간 결과를 저장하고 읽는 비용을 줄여 실제 wall-clock time과 memory 사용량을 개선한다.

## 학습과 추론에서의 의미

FlashAttention은 학습과 추론 모두에서 의미가 있다.

학습에서는 activation memory를 줄이고 attention 연산 속도를 높일 수 있다. 긴 sequence로 학습할 때 GPU memory 부족을 완화하는 데 도움이 된다.

추론에서는 prompt prefill 단계에서 특히 중요하다. LLM inference는 크게 두 단계로 볼 수 있다.

- Prefill: 입력 prompt 전체를 한 번에 처리해 KV cache를 만든다.
- Decode: 다음 token을 하나씩 생성한다.

FlashAttention은 여러 token을 한 번에 처리하는 attention 계산에서 효과가 크기 때문에 prefill 단계에서 자주 중요해진다.

반면 decode 단계에서는 한 번에 새 token 하나를 생성하는 경우가 많다. 이때는 FlashAttention보다 KV cache 관리, batching, memory bandwidth 같은 요소가 더 큰 병목이 될 수 있다.

그래서 LLM serving을 볼 때는 FlashAttention만 따로 보기보다 vLLM의 continuous batching, PagedAttention, KV cache 최적화와 함께 봐야 한다.

## FlashAttention-2

FlashAttention-2는 FlashAttention의 아이디어를 유지하면서 GPU 병렬성을 더 잘 활용하도록 개선한 버전이다.

FlashAttention-1이 memory IO를 줄이는 데 집중했다면, FlashAttention-2는 다음 부분을 더 개선했다.

- non-matmul 연산 감소
- thread block 간 work partitioning 개선
- warp 간 shared memory read/write 감소
- 더 높은 GPU occupancy 달성

즉, FlashAttention-2는 같은 IO-aware attention이라도 GPU 내부에서 일을 나누는 방식을 더 효율적으로 만든 것이다.

실제 LLM 학습과 추론 라이브러리에서는 FlashAttention-2 계열 구현을 사용하는 경우가 많다.

## FlashAttention-3

FlashAttention-3는 H100 같은 Hopper GPU 아키텍처를 더 잘 활용하기 위한 버전이다.

핵심 방향은 다음과 같다.

- Tensor Core 연산과 data movement를 더 잘 겹치기
- TMA 같은 Hopper 기능 활용
- FP8 같은 low-precision 연산 지원
- 비동기 실행으로 GPU pipeline 효율 높이기

즉, FlashAttention-3는 FlashAttention의 기본 아이디어를 유지하면서 최신 GPU 하드웨어에 더 맞게 최적화한 버전으로 볼 수 있다.

다만 모든 환경에서 FlashAttention-3를 바로 쓰는 것은 아니다. GPU 아키텍처, CUDA 버전, 프레임워크 지원 여부에 따라 사용할 수 있는 구현이 달라진다.

## PyTorch에서의 사용

요즘은 직접 FlashAttention kernel을 호출하지 않아도, PyTorch나 Hugging Face Transformers 내부에서 optimized attention backend를 사용하는 경우가 많다.

PyTorch에는 `scaled_dot_product_attention` API가 있고, 환경에 따라 math, memory efficient attention, flash attention backend 중 하나를 사용할 수 있다.

개념적으로는 다음과 같은 형태다.

```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=attn_mask,
    dropout_p=0.0,
    is_causal=True,
)
```

다만 실제로 어떤 backend가 선택되는지는 GPU, dtype, tensor shape, PyTorch 버전에 따라 달라질 수 있다. 성능을 확인하려면 단순히 코드가 실행되는지만 보지 말고 profiler나 benchmark로 실제 kernel을 확인해야 한다.

## Trade-off

FlashAttention은 매우 유용하지만, 모든 문제를 해결하는 마법은 아니다.

먼저 GPU kernel 수준의 최적화이기 때문에 하드웨어와 소프트웨어 환경의 영향을 많이 받는다. 특정 GPU, CUDA, PyTorch 조합에서는 기대한 backend가 선택되지 않을 수 있다.

또한 attention 계산 자체가 사라지는 것은 아니다. sequence length가 길어질수록 여전히 계산량은 커진다. FlashAttention은 memory IO를 줄여 더 효율적으로 만들지만, quadratic attention의 근본적인 조합 수는 그대로 남는다.

마지막으로 decode 중심 workload에서는 attention kernel보다 KV cache memory, scheduler, batching이 더 큰 병목일 수 있다. 그래서 serving에서는 FlashAttention과 PagedAttention을 구분해서 이해해야 한다.

## FlashAttention과 PagedAttention 차이

둘 다 LLM serving 글에서 자주 등장하지만 해결하는 문제가 다르다.

FlashAttention은 attention 계산 kernel을 빠르고 memory-efficient하게 만드는 방법이다.

PagedAttention은 vLLM에서 KV cache memory를 block 단위로 관리해 serving 중 memory 낭비를 줄이는 방법이다.

정리하면 다음과 같다.

| 항목 | FlashAttention | PagedAttention |
| --- | --- | --- |
| 주 대상 | attention 계산 | KV cache 관리 |
| 핵심 문제 | HBM read/write 비용 | 동시 요청의 cache memory 낭비 |
| 주 효과 | attention 속도와 memory 개선 | serving throughput과 memory utilization 개선 |
| 관련 단계 | prefill, training에서 특히 중요 | decode와 multi-request serving에서 중요 |

둘은 경쟁 관계라기보다 서로 다른 병목을 해결하는 기술이다.

## 정리

FlashAttention은 Transformer attention을 IO-aware하게 다시 구현한 방법이다.

핵심은 다음과 같다.

- 기존 attention은 `N x N` attention matrix를 만들기 때문에 memory IO 비용이 크다.
- FlashAttention은 tiling으로 `Q`, `K`, `V`를 block 단위로 처리한다.
- attention matrix 전체를 HBM에 저장하지 않고 output을 누적한다.
- online softmax를 사용해 block 단위 계산에서도 정확한 softmax 결과를 유지한다.
- attention의 `O(N^2)` 계산 자체를 없애는 것은 아니지만, 실제 GPU memory 사용량과 실행 시간을 크게 줄인다.

LLM을 다룰 때 FlashAttention은 long context, prefill, 학습 throughput을 이해하는 데 중요한 개념이다.

다만 serving 전체 성능은 FlashAttention 하나만으로 결정되지 않는다. 실제 운영에서는 KV cache, batching, scheduler, quantization, tensor parallelism까지 함께 봐야 한다.

## 참고 자료

- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- Jay Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
