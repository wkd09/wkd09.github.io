---
title: "AWQ와 SmoothQuant 논문 정리: activation outlier를 다루는 두 가지 PTQ 방법"
date: 2026-06-17 00:00:00 +0900
categories:
  - research
tags:
  - AI
  - LLM
  - Quantization
  - PTQ
  - Paper
source: "arXiv - AWQ, SmoothQuant"
---

# AWQ와 SmoothQuant

> SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models  
> Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han  
> arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)]

> AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration  
> Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han  
> arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00978)]

## 한 줄 정의

SmoothQuant와 AWQ는 모두 LLM post-training quantization에서 **activation outlier가 만든 양자화 난이도를 channel-wise scaling으로 완화하는 방법**이다.

다만 목표는 다르다.

- SmoothQuant는 weight와 activation을 모두 INT8로 만드는 **W8A8 quantization**을 목표로 한다.
- AWQ는 activation은 고정밀로 두고 weight를 4bit 수준으로 줄이는 **weight-only quantization**을 목표로 한다.

즉 둘 다 activation statistics를 사용하지만, SmoothQuant는 activation quantization을 가능하게 만들고, AWQ는 weight quantization error를 줄인다.

![SmoothQuant 원문 Figure: activation outlier가 weight로 이동하는 방식](/assets/images/blog/smoothquant-paper-figure.png)

*원 논문 Figure 4는 SmoothQuant가 activation outlier를 weight 쪽으로 옮겨 quantization difficulty를 줄이는 모습을 보여준다.*

![AWQ 원문 Figure: salient weights 보호와 per-channel scaling](/assets/images/blog/awq-paper-figure.png)

*원 논문 Figure 2는 salient weight를 보호하거나 scaling하는 방식이 INT3 quantization 성능을 어떻게 바꾸는지 보여준다.*

## 왜 필요한가

LLM inference에서 quantization을 하는 이유는 단순하다.

- 모델 파라미터 저장 메모리를 줄인다.
- memory bandwidth 병목을 줄인다.
- INT8, INT4 같은 low precision kernel을 활용해 throughput을 높인다.

하지만 LLM은 작은 CNN이나 BERT류 모델보다 quantization이 어렵다. 특히 activation에 큰 outlier가 생긴다.

uniform quantization을 생각해보면 문제는 더 명확하다.

$$
\bar{x} = round(x / \Delta)
$$

여기서 quantization scale인 $\Delta$는 보통 tensor의 최대 절댓값에 영향을 받는다. activation channel 중 일부에 매우 큰 값이 있으면 전체 범위가 그 outlier에 맞춰진다. 그러면 대부분의 작은 값들은 쓸 수 있는 quantization level이 줄어들어 정보가 뭉개진다.

즉 LLM quantization의 핵심 문제는 다음과 같다.

> weight는 상대적으로 quantize하기 쉽지만, activation outlier는 quantization range를 크게 만들어 정확도를 망가뜨린다.

SmoothQuant와 AWQ는 이 문제를 정면으로 다룬다.

## SmoothQuant 핵심 아이디어

SmoothQuant의 목표는 모든 주요 matrix multiplication을 INT8로 실행하는 것이다. 이를 위해서는 weight뿐 아니라 activation도 INT8로 양자화해야 한다.

문제는 activation outlier다. SmoothQuant는 activation을 직접 per-channel quantization하지 않는다. per-channel activation scale은 일반적인 INT8 GEMM kernel과 잘 맞지 않기 때문이다.

대신 linear layer의 등가 변환을 사용한다.

$$
Y = XW
$$

$$
Y = (X diag(s)^{-1})(diag(s)W)
$$

여기서 $s$는 input channel별 smoothing factor이다.

이 변환은 수학적으로 원래 linear layer와 같다. activation 쪽에서는 outlier channel을 $s$로 나누어 부드럽게 만들고, weight 쪽에는 같은 scale을 곱해 출력을 보존한다.

결과적으로 quantization difficulty를 activation에서 weight로 일부 옮긴다.

SmoothQuant에서 중요한 점은 weight가 activation보다 quantization에 강하다는 관찰이다. activation outlier를 그대로 INT8로 만들면 정확도가 크게 떨어지지만, 그 부담을 weight로 옮기면 weight는 상대적으로 잘 버틴다.

## SmoothQuant의 smoothing factor

논문은 migration strength를 조절하기 위해 $\alpha$를 사용한다.

$$
s_j = \frac{max(|X_j|)^\alpha}{max(|W_j|)^{1-\alpha}}
$$

- $\alpha$가 크면 activation outlier를 더 많이 weight 쪽으로 옮긴다.
- $\alpha$가 작으면 weight 쪽 부담을 줄이고 activation 쪽에 더 남긴다.

극단적으로 activation을 너무 많이 줄이면 weight에 outlier가 생겨 weight quantization error가 커진다. 반대로 activation을 충분히 줄이지 않으면 activation quantization error가 남는다.

따라서 SmoothQuant는 activation과 weight 사이에서 quantization difficulty를 적절히 나누는 방법이다.

## AWQ 핵심 아이디어

AWQ는 SmoothQuant와 문제의 초점이 다르다. AWQ는 activation을 INT4나 INT8로 낮추는 것이 아니라, weight를 낮은 bit로 줄이는 weight-only quantization을 목표로 한다.

weight-only quantization에서는 보통 activation은 FP16/BF16으로 유지하고, weight만 INT4 같은 낮은 precision으로 저장한다. LLM serving에서 weight memory와 bandwidth를 줄이는 데 효과적이고, 특히 작은 batch나 edge device에서 유리하다.

AWQ의 핵심 관찰은 다음과 같다.

> 모든 weight가 똑같이 중요하지 않다. activation이 크게 들어오는 channel의 weight가 quantization error에 더 민감하다.

linear layer를 다시 보면:

$$
Y = XW
$$

weight quantization error가 $\Delta W$라고 할 때 output error는 대략 다음처럼 activation과 함께 증폭된다.

$$
\Delta Y \approx X \Delta W
$$

즉 weight 자체의 크기만 보고 중요한 channel을 고르면 부족하다. 실제 inference에서 그 weight가 얼마나 큰 activation과 곱해지는지가 중요하다.

그래서 AWQ는 calibration data로 activation statistics를 모으고, activation 관점에서 중요한 weight channel을 찾는다.

## Salient weight를 어떻게 보호하는가

가장 단순한 방법은 중요한 weight 일부를 FP16으로 남기는 것이다. 하지만 이는 mixed precision이 되어 kernel 구현이 복잡해지고 hardware efficiency가 떨어진다.

AWQ는 이 방향을 피한다. 논문은 salient weight를 따로 고정밀로 저장하는 대신, 중요한 channel을 scale up하는 등가 변환을 사용한다.

개념적으로는 다음과 같다.

$$
Y = XW = (X diag(s)^{-1})(diag(s)W)
$$

중요한 channel의 weight를 scale up하면 quantization grid에서 상대적으로 더 잘 표현된다. 그 대신 activation 쪽에 inverse scale을 적용해 원래 출력을 유지한다.

이 변환 자체는 원래 layer와 등가지만, quantization 이후에는 차이가 생긴다. scale up된 salient channel은 quantization error가 줄어들고, 전체 output error도 작아진다.

AWQ가 중요한 이유는 이 과정을 backpropagation이나 layer reconstruction 없이 수행한다는 점이다. calibration data로 activation statistics를 수집하고, scale을 탐색한 뒤 weight를 quantize한다.

## SmoothQuant와 AWQ 비교

| 항목 | SmoothQuant | AWQ |
| --- | --- | --- |
| 목표 | W8A8 quantization | low-bit weight-only quantization |
| 주된 bit-width | INT8 weight, INT8 activation | INT4 weight, FP16/BF16 activation 계열 |
| 핵심 문제 | activation outlier 때문에 INT8 activation이 어려움 | 중요한 weight channel의 quantization error가 output에 크게 반영됨 |
| activation statistics 역할 | activation outlier channel을 찾고 smoothing factor 계산 | salient weight channel을 찾는 기준 |
| scaling 방향 | activation을 smooth하게 만들고 weight로 난이도 이전 | salient weight channel을 scale up해 quantization error 감소 |
| runtime 목표 | INT8 GEMM 활용 | weight memory/bandwidth 절감과 efficient low-bit kernel |
| 학습 필요 여부 | training-free PTQ | backpropagation 없는 PTQ |

두 방법은 같은 수식 형태를 공유한다.

$$
XW = (X diag(s)^{-1})(diag(s)W)
$$

하지만 이 등가 변환을 쓰는 목적이 다르다.

SmoothQuant는 activation을 quantize하기 쉽게 만들기 위해 $X$의 channel range를 줄인다. AWQ는 중요한 weight channel이 low-bit grid에서 덜 손상되도록 $W$ 쪽을 보호한다.

## 직관적 설명

SmoothQuant는 activation outlier를 "weight 쪽으로 이사"시키는 방법이다.

예를 들어 activation의 어떤 channel만 값이 유난히 크면, INT8 quantization에서는 그 channel 때문에 전체 scale이 커진다. SmoothQuant는 그 channel을 미리 나누어 activation 범위를 평평하게 만든다. 대신 weight에 같은 만큼 곱해 수학적 출력은 유지한다.

AWQ는 "자주 세게 쓰이는 weight channel에 더 좋은 해상도를 주는" 방법이다.

어떤 weight가 작아 보여도, 그 weight에 곱해지는 activation이 크면 output에 미치는 영향은 커진다. AWQ는 이런 channel을 activation statistics로 찾아서, quantization 전에 scale을 키운다. 그러면 4bit grid에서도 중요한 weight가 덜 손상된다.

## 주요 결과 해석

SmoothQuant 논문은 W8A8 quantization으로 큰 LLM에서도 정확도 손실을 작게 유지하면서 메모리 사용량과 inference cost를 줄일 수 있음을 보인다. arXiv abstract 기준으로 최대 1.56배 speedup과 2배 memory reduction을 보고하고, 530B 규모 모델을 single node에서 serving할 수 있다고 설명한다.

AWQ 논문은 salient weight의 약 1%만 잘 보호해도 quantization error를 크게 줄일 수 있다고 주장한다. 또한 backpropagation이나 reconstruction 없이 calibration statistics만 사용하기 때문에 instruction-tuned LM, domain-specific benchmark, multi-modal LM에도 일반화가 좋다고 설명한다. 함께 제안한 TinyChat은 kernel fusion과 weight packing을 통해 Hugging Face FP16 구현 대비 3배 이상의 speedup을 보고한다.

여기서 중요한 해석은 "algorithm만으로는 충분하지 않다"는 점이다.

두 논문 모두 hardware-friendly quantization을 강조한다. 실제 serving에서 중요한 것은 이론적 bit-width만이 아니라, 해당 quantization format이 GPU kernel, memory layout, dequantization overhead와 잘 맞는지이다.

## 실제 시스템과의 연결

LLM serving 관점에서 두 방법은 쓰임새가 다르다.

SmoothQuant는 cloud/server inference에서 INT8 GEMM을 활용하고 싶을 때 자연스럽다. batch가 어느 정도 있고, activation까지 INT8로 낮춰 compute를 가속할 수 있는 환경에서 의미가 크다.

AWQ는 weight memory가 병목인 환경에서 자연스럽다. 특히 on-device, edge GPU, 작은 batch inference처럼 weight bandwidth가 중요한 상황에서 weight-only quantization은 실용적이다.

실제 시스템에서는 다음 질문으로 선택할 수 있다.

- activation까지 INT8로 낮춰 GEMM 자체를 빠르게 만들고 싶은가? 그러면 SmoothQuant 계열이 맞다.
- activation은 고정밀로 두더라도 weight 저장량과 bandwidth를 크게 줄이고 싶은가? 그러면 AWQ 계열이 맞다.
- batch가 큰 server serving인가, 작은 batch의 local inference인가?
- 사용하는 inference engine이 해당 quantization kernel을 효율적으로 지원하는가?

## 한계점

SmoothQuant의 한계는 activation과 weight 사이의 migration strength를 잘 잡아야 한다는 점이다. 너무 많이 옮기면 weight quantization이 어려워지고, 너무 적게 옮기면 activation quantization 문제가 남는다. 또한 실제 speedup은 INT8 kernel 지원, operator fusion, framework integration에 크게 의존한다.

AWQ의 한계는 weight-only quantization이므로 activation memory와 activation compute를 직접 줄이지 않는다는 점이다. 또한 low-bit weight를 실제로 빠르게 쓰려면 dequantization overhead가 작아야 하고, kernel과 packing format이 중요하다. algorithm이 좋아도 runtime이 받쳐주지 않으면 wall-clock speedup은 제한된다.

두 방법 모두 calibration data에 의존한다. 학습은 하지 않지만, activation statistics를 추정해야 하므로 calibration sample이 실제 inference distribution과 너무 다르면 최적 scale이 덜 맞을 수 있다.

## 내가 이해한 점

SmoothQuant와 AWQ는 모두 "LLM quantization에서는 소수의 activation outlier가 전체 quantization range를 망가뜨릴 수 있다"는 사실에서 출발한다.

하지만 해결 방향은 다르다.

SmoothQuant는 activation outlier를 줄여서 activation도 INT8로 만들 수 있게 한다. 그래서 목표는 W8A8과 INT8 GEMM이다.

AWQ는 activation outlier를 직접 quantize하지 않는다. 대신 activation이 큰 channel에 연결된 weight가 더 중요하다고 보고, 그 weight channel을 4bit quantization에서 덜 망가지게 만든다. 그래서 목표는 W4A16 계열 weight-only inference다.

정리하면 다음과 같다.

> SmoothQuant는 activation quantization을 가능하게 만들기 위한 smoothing이고, AWQ는 weight-only quantization에서 중요한 weight channel을 보호하기 위한 smoothing이다.

이 차이를 알고 보면 두 논문은 서로 경쟁하는 방법이라기보다, 서로 다른 serving 조건을 겨냥한 quantization 설계로 이해하는 편이 맞다.
