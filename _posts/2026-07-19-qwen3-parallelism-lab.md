---
title: "Qwen3-4B로 확인한 LLM 병렬화 실습: LoRA, DDP, FSDP, TP, vLLM"
date: 2026-07-19 00:00:00 +0900
last_modified_at: 2026-08-31 00:00:00 +0900
categories:
  - engineering
tags:
  - LLM
  - Qwen
  - Distributed Training
  - DDP
  - FSDP
  - Tensor Parallelism
  - vLLM
---

LLM 병렬화는 이름만 보면 비슷하지만 해결하는 문제는 서로 다르다. DDP는 학습 시간을 줄이고, FSDP는 GPU 한 장이 감당해야 하는 training memory를 줄이며, Tensor Parallelism(TP)은 한 model의 계산을 여러 GPU에 나눈다. Serving에서는 vLLM의 batching과 병렬화 구성이 throughput과 latency의 균형을 바꾼다.

이번 글에서는 `Qwen/Qwen3-4B`로 진행한 실습 결과를 기준으로, 단일 GPU LoRA부터 2-GPU DDP, FSDP, TP, vLLM 서빙까지 한 흐름으로 정리한다. 개념 자체는 이전 글에서 다뤘으므로, 여기서는 **어떤 병목이 실제로 드러났고 어떤 수치로 확인했는지**에 집중한다.

## 실습 구성

학습 실습에는 256개의 짧은 대화형 예제와 BF16을 사용했다. 모델은 모두 Qwen3-4B이며, LoRA 실습에서는 rank 16, `lora_alpha=32`, dropout 0.05를 적용하고 모든 linear layer를 대상으로 삼았다.

| 단계 | 목적 | 확인한 것 |
| --- | --- | --- |
| Single-GPU LoRA | 기준선 만들기 | 시간과 피크 VRAM |
| 2-GPU DDP LoRA | 데이터 병렬화 | 처리 시간, 파라미터 동기화 |
| DDP full fine-tuning / FSDP | 메모리 한계 확인 | OOM 여부와 shard 후 메모리 |
| TP=2 MLP | 모델 내부 분할 검증 | 파라미터 분할과 수치 일치 |
| vLLM serving | 동시 요청 처리 | TTFT, 지연 시간, 처리량 |

## 1. 기준선: 단일 GPU LoRA

가장 먼저 단일 GPU에서 LoRA SFT를 실행했다. `per_device_train_batch_size=2`, gradient accumulation 4, 3 epoch, 최대 길이 512로 설정했다. 따라서 한 optimizer update에 반영되는 유효 batch는 8이다.

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
)

training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True,
)
```

| 지표 | 결과 |
| --- | ---: |
| 학습 시간 | 116.38 s |
| 평균 step 시간 | 1.212 s |
| 피크 VRAM | 9.27 GiB |
| train loss | 0.3218 |
| 학습 샘플 처리량 | 6.62 samples/s |

4B 모델 전체를 업데이트하지 않고 adapter만 학습하므로, 단일 GPU에서도 약 9.3 GiB로 실행할 수 있었다. 이 수치가 다음 DDP 실습의 기준선이다.

## 2. 2-GPU DDP로 LoRA 학습 시간 줄이기

같은 LoRA 설정을 두 GPU에서 실행했다. 각 프로세스는 서로 다른 데이터 shard를 처리하고, backward 과정에서 학습 가능한 LoRA 파라미터의 gradient를 동기화한다. 유효 global batch는 `2 × 4 × 2 = 16`이 된다.

실습 뒤에는 각 rank의 trainable parameter 합계를 checksum으로 모아, 두 모델 replica가 같은 값으로 끝났는지도 확인했다.

| 지표 | 단일 GPU | 2-GPU DDP |
| --- | ---: | ---: |
| 전체 학습 시간 | 116.38 s | 61.17 s |
| 학습 샘플 처리량 | 6.62 samples/s | 12.74 samples/s |
| 피크 VRAM/GPU | 9.27 GiB | 9.39 GiB |
| adapter checksum rank 0 / 1 | - | 19.679420 / 19.679420 |
| checksum 차이 | - | 0.0 |

전체 학습 시간은 약 **1.90배** 빨라졌고, 샘플 처리량도 거의 두 배가 됐다. 반면 GPU 한 장이 쓰는 VRAM은 줄지 않았다. 이것이 DDP의 핵심 trade-off다. DDP는 모델을 GPU마다 **복제**하므로 시간을 줄이는 데 적합하지만, 모델과 optimizer state가 한 장에 들어가지 않는 문제를 해결하지는 못한다.

여기서 loss 수치를 단순 비교하면 안 된다. DDP는 global batch가 16이라 같은 데이터를 처리하면서 optimizer update 횟수가 단일 GPU 실험과 달라진다. 이 실험의 목적은 수렴 성능 비교가 아니라, 같은 데이터 규모에서 병렬 학습과 동기화가 제대로 동작하는지와 처리 시간 변화를 확인하는 것이었다.

## 3. Full fine-tuning에서 DDP가 막힌 지점

다음으로 LoRA가 아닌 Qwen3-4B 전체 파라미터를 2-GPU DDP로 학습해 보았다. 모델 파라미터만 각 GPU에 복제해도 크지만, full fine-tuning에서는 gradient와 AdamW optimizer state까지 추가된다.

결과는 `AdamW optimizer.step()`에서 OOM이었다.

| 방식 | 결과 | 관측 메모리 |
| --- | --- | ---: |
| 2-GPU DDP full fine-tuning | OOM at `AdamW optimizer.step` | 43.76 GiB |

DDP의 각 rank에는 약 40.22억 개 전체 파라미터가 그대로 존재한다. 따라서 GPU가 두 장이라는 사실은 전체 모델을 두 조각으로 나눈다는 뜻이 아니다. 각 GPU는 완전한 모델 replica와 학습 상태를 보관한다.

## 4. FSDP FULL_SHARD로 full fine-tuning 실행하기

FSDP에서는 파라미터, gradient, optimizer state를 rank마다 shard한다. 이 실습에서는 Qwen3 decoder layer를 wrapping 단위로 지정하고, `FULL_SHARD` 및 BF16 mixed precision을 적용했다.

```python
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=device,
    limit_all_gathers=True,
)

# optimizer는 FSDP 적용 뒤에 생성해야 state도 shard된다.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

| 지표 | FSDP FULL_SHARD 결과 |
| --- | ---: |
| 전체 파라미터 | 4,022,468,096 |
| rank당 파라미터 shard | 2,011,234,048 |
| wrapping 직후 resident memory | 3.75 GiB |
| 학습 중 피크 VRAM | 18.78 GiB |
| 20 step 실행 시간 | 27.14 s |
| 처리량 | 2.95 samples/s |
| 마지막 loss | 0.1629 |

DDP에서는 optimizer step까지 가지 못했지만, FSDP는 2개 rank에 각각 약 절반의 parameter shard를 둬 20 step을 끝까지 수행했다. 다만 FSDP가 메모리를 "공짜로" 줄이는 것은 아니다. 각 layer를 계산할 때 필요한 parameter shard를 all-gather하고, 계산이 끝나면 다시 shard하므로 통신과 구현 복잡도가 늘어난다. 메모리가 병목일 때 그 비용을 감수하는 선택이다.

## 5. Tensor Parallelism: MLP를 실제로 나누어 보기

TP 실습에서는 Qwen3 첫 번째 decoder layer의 MLP만 분리해 TP=2로 실행했다. hidden size는 2,560이고, MLP 파라미터 74,711,040개를 두 GPU에 절반씩 배치했다.

```python
tp_mlp = parallelize_module(
    tp_mlp,
    tp_mesh,
    {
        "gate_proj": ColwiseParallel(),
        "up_proj": ColwiseParallel(),
        "down_proj": RowwiseParallel(),
    },
)
```

`gate_proj`와 `up_proj`는 output feature 방향으로 나누면 각 GPU가 자기 projection 결과를 독립적으로 만들 수 있다. 반대로 `down_proj`는 input feature 방향으로 나뉜 partial result를 합쳐야 하므로 all-reduce가 필요하다. 이 조합이 Transformer MLP에 자주 사용되는 TP 패턴이다.

| 지표 | 결과 |
| --- | ---: |
| TP 크기 | 2 |
| 전체 MLP 파라미터 | 74,711,040 |
| rank별 로컬 파라미터 | 37,355,520 / 37,355,520 |
| 최대 절대 오차 | 0.25 |
| 상대 RMSE | 0.00273 |
| 허용 상대 RMSE | 0.01 |
| BF16 허용 오차 내 일치 | true |

최대 절대 오차만 보면 0.25가 커 보일 수 있다. 하지만 BF16 연산과 collective communication에서는 출력 크기에 대한 상대 오차가 더 유용하다. 상대 RMSE가 0.01보다 충분히 작았으므로, TP 결과가 기준 MLP 출력과 BF16 허용 범위에서 일치함을 확인했다.

이 실습은 TP가 **MLP 하나를 정확히 분할할 수 있는지** 검증한 것이며, 전체 모델의 end-to-end 속도 향상을 측정한 벤치마크는 아니다. 전체 Transformer에 TP를 적용하면 attention, embedding, loss 계산과 통신 패턴까지 함께 설계해야 한다.

## 6. vLLM 서빙: 동시성에 따라 달라지는 처리량과 지연 시간

마지막으로 vLLM의 OpenAI 호환 `/v1/chat/completions` endpoint에 같은 짧은 요청을 동시에 보냈다. 각 요청은 최대 64 토큰을 생성하고, warm-up 이후 streaming 요청으로 TTFT(Time To First Token)를 한 번 측정했다. 이후 비스트리밍 요청의 평균 latency, p50, 처리량을 기록했다.

아래 표는 저장된 실험 결과 중 동시성 32, 128, 256에서의 결과다. `TP`, `PP`, `DP` 표기는 해당 vLLM 서버 실행 구성을 구분하는 label이다.

| 구성 | 동시성 | TTFT | 평균 latency | completion tokens/s | requests/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP=1 | 32 | 43.0 ms | 1.386 s | 1,463.7 | 22.87 |
| TP=2 | 128 | 34.8 ms | 2.156 s | 3,663.1 | 57.24 |
| TP=2 | 256 | 32.4 ms | 2.248 s | 6,877.3 | 107.46 |
| PP=2 | 32 | 37.5 ms | 1.760 s | 1,154.2 | 18.03 |
| PP=2 | 128 | 30.4 ms | 1.675 s | 4,727.1 | 73.86 |
| PP=2 | 256 | 24.9 ms | 2.005 s | 7,789.6 | 121.71 |
| DP=2 | 32 | 56.1 ms | 1.485 s | 1,247.3 | 19.49 |
| DP=2 | 128 | 32.4 ms | 1.443 s | 5,439.3 | 84.99 |
| DP=2 | 256 | 30.2 ms | 1.733 s | 9,035.3 | 141.18 |

이 결과에서 특히 눈에 띈 점은 동시성을 높일수록 처리량이 크게 증가한다는 것이다. 예를 들어 DP=2 구성은 concurrency 32에서 1,247.3 tokens/s였지만, concurrency 256에서는 9,035.3 tokens/s를 기록했다. 개별 요청의 평균 latency는 늘어나지만, continuous batching이 GPU를 더 촘촘히 사용하면서 서비스 전체 처리량이 좋아진 것으로 볼 수 있다.

다만 이 표만으로 "DP가 항상 TP나 PP보다 빠르다"고 결론 내릴 수는 없다. 모델 크기, GPU 연결 방식, prompt와 output 길이, replica 수, server option, traffic 형태가 모두 결과에 영향을 준다. 특히 이번 결과는 TTFT를 한 번 측정한 값이므로, 운영 결정을 위해서는 충분한 반복 측정과 p95/p99 latency도 추가로 수집해야 한다.

## 내가 이해한 핵심: 병렬화 방식은 병목에 맞춰 선택한다

이번 실습에서 얻은 결론은 단순하다.

- **LoRA + DDP**는 GPU당 메모리가 충분할 때 학습 시간을 줄이는 좋은 출발점이다. 이 실습에서는 2-GPU DDP가 약 1.90배 빠르게 끝났다.
- **FSDP**는 full fine-tuning처럼 DDP replica가 메모리에 들어가지 않을 때 필요하다. DDP가 OOM 난 4B 모델을 FSDP FULL_SHARD로는 실행할 수 있었다.
- **TP**는 한 GPU에 모델의 계산 단위를 모두 올리기 어려울 때 모델 내부 matrix operation을 나누는 방법이다. 대신 각 layer의 통신 비용을 함께 고려해야 한다.
- **vLLM serving**에서는 단일 요청 latency만이 아니라 동시성별 tokens/s, requests/s, TTFT와 tail latency를 같이 봐야 한다.

결국 "GPU가 여러 장이니 병렬화를 쓰자"가 아니라, 먼저 병목이 **학습 시간인지, GPU 메모리인지, 한 요청의 지연 시간인지, 전체 서빙 처리량인지**를 정해야 한다. 그 질문에 따라 DDP, FSDP, TP, DP serving의 선택이 달라진다.

## 실습 코드와 재현 시 확인할 점

실습 코드는 다음 파일 단위로 구성했다.

- `step1-single-lora/train_single_lora.py`: 단일 GPU LoRA 기준선
- `step2-ddp-lora/train_ddp_lora.py`: 2-GPU DDP LoRA와 checksum 검증
- `step3-fsdp/train_ddp_full.py`, `train_fsdp.py`: full fine-tuning의 DDP OOM 및 FSDP 비교
- `step4-tp/tp_qwen3_mlp.py`: Qwen3 MLP TP=2 수치 검증
- `step5-vllm-serving/bench_vllm.py`: OpenAI 호환 endpoint 부하 측정

재현 실험에서는 다음 조건을 고정하는 것이 중요하다.

- GPU 모델과 GPU 간 연결 방식
- PyTorch, Transformers, vLLM 및 CUDA 버전
- model revision, dtype, 최대 sequence length
- batch size, gradient accumulation, optimizer와 learning rate
- 서빙 요청의 prompt 길이, 생성 토큰 수, concurrency, warm-up 횟수

이 조건이 바뀌면 숫자는 달라질 수 있지만, 각 병렬화 방식이 해결하는 병목과 trade-off는 그대로 남는다.
