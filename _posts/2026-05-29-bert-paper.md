---
title: "BERT 논문 정리: 양방향 Transformer Encoder로 언어 이해하기"
date: 2026-05-29 01:20:00 +0900
categories:
  - research
tags:
  - AI
  - NLP
  - BERT
  - Paper
source: "Notion PDF Export - BERT"
---

# Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding

> arXiv 2019 [[paper](https://arxiv.org/pdf/1810.04805)]
> 

> Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
> 

> Google AI Language
> 

### 연구 배경 및 동기

- 기존 언어 모델의 한계: 기존 모델(ELMo, GPT 등등) 주로 **단방향** 언어 모델로 사전 학습을 하였다. 특히 GPT와 같은 미세조정 기반 방식은 self attention에서 각 토큰이 이전 토큰에만 의존하도록 제한된다.
- 양방향 컨텍스트의 중요성: 단방향 제약에서 질문과 응답 같이 양방향 컨텍스트를 통합하는 것이 중요한 토큰 작업에 안 좋다.

### 핵심 아이디어

- 양방향 인코더 표현: Bert는 **B**idirectional **E**ncoder **R**epresentation from **T**ransformers의 약자로 왼쪽 및 오른쪽 컨텍스트 모두를 심층 양방향 표현을 사전 학습하는 새로운 언어 모델이다.
- 마스크 언어 모델(MLM): 입력 토큰중 일부를 무작위로 마스킹하고, 마스킹된 단어를 예측하는 것이다.
- 다음 문장 예측(NSP): 문장 간 관계 이해를 위한 추가적인 사전학습 태스크 사용
- Bert는 단어 생성보다, 복잡한 자연어 구조와 의미를 이해하는 모델

### 방법론

- 대략적인 방법: 두가지 단계로 사전 학습과 미세 조정으로 구성된다. 사전 학습 동안 모델은 레이블 없는 데이터로 다양한 사전학습 태스크를 학습하고, 미세 조정시 사전 학습된 매개변수로 초기화된 Bert로 다운스트림 태스크의 레이블있는 데이터로 모든 매개 변수를 조정한다.

## Embedding

![bert embedding](/assets/images/blog/bertembedding.png)

1. Token Embeddings: WordPiece embeddings
- BPE와 유사한 subword-based tokenization
    
    :텍스트를 단어가 아닌 subword 단위로 분할, 빈도가 높을수록 그 자체로 vocab에 저장되고 빈도가 적을수록 subword로 분할되어 vocab에 저장
    
- BPE는 텍스트에서 가장 빈번한 문자쌍을 반복적으로 병합하는 단순 빈도 기반 방식
- WordPiece는 언어 모델의 성능 향상을 최대화하는 방향으로 문자쌍을 결합
1. Segment Embedding
- 여러 sentence를 연결하여 입력값으로 사용하는 경우, 두 sentence 사이에 구분 토큰 [SEP] 추가
- [SEP] 토큰을 기준으로 각 토큰이 어떤 sentence에 속하는지 알려주는 embedding
- 입력의 첫번째 [SEP]까지는 Segment embedding A (보통 0),
    
    그 이후부터 다음 [SEP]까지는 Segment embedding B (보통 1)로 설정 
    
1. Position Embedding
- transformer encoder에서 self-attention을 사용하기 때문에 추가해주는 위치 정보 embedding
- sin/cos 함수가 아닌, 모델이 학습과정에서 최적화하는 각 위치별 고유한 벡터값을 사용
- Position embedding 벡터 종류 = 512개
    
    BERT에서는 문장의 최대 길이를 512로 하고 있어 512개의 서로 다른 위치가 존재하게 됨
    

## Pre-training

![bert pretraining concept](/assets/images/blog/bertscore.png)

1. Masked Language Model, MLM
- 입력에서 무작위하게 15%의 토큰을 마스킹하고 주변 단어를 사용해 마스킹된 토큰 예측
- Bert가 단어의 양방향 문맥, 단어 간 관계, 문법이나 문장 구조등을 학습
- 마스킹된 15%들의 구성
    - 80%: 토큰을 [MASK]로 변경
        
        Ex) My dog is cute. he likes playing → My [MASK] is cute. he likes playing
        
    - 10%: 토큰을 무작위로 다른 토큰으로 변경
        
        Ex) My dog is cute. he likes playing → My dog is cute. King likes playing
        
    - 10%: 동일한 토큰 그대로 남겨둠
        
        Ex) My dog is cute. he likes playing → My dog is cute. he likes playing
        

다른 토큰으로 변경된 10%와 그대로 남겨둔 10% 모두 모델은 단어가 무엇인지 예측.

**[MASK]**는 fine-tuning에서 사용되지 않는 토큰 = 실제 데이터셋에서는 나타나지 않는 토큰

→ 모델이 [MASK] 토큰만 잘 예측하게 학습하는 것이 아닌 실제 단어를 잘 예측하도록 다른 토큰과 그대로 남겨둬야된다. pre-training과 fine-tuning간 입력 토큰의 불일치 문제를 해결하면서 모델의 언어 이해력을 더 향상

1. Next Sentence Prediction, NSP
- 두 문장 A, B가 이어지는 문장인지 아닌지를 예측
- sentence relationships를 학습

- 문장 쌍 데이터셋을 아래와 같이 구성
    - 50% : A와 B가 관련된 문장, A뒤에 B가 따라오는 문장, 'IsNext'라고 labeling
    - 50% : A와 B가 관련없는 문장, 무작위 선택, 'NotNext'라고 labeling
- [CLS] 토큰의 final hidden vector C를 이용해 이진 분류
- C를 feed-forward network → softmax 순서로 통과 시켜 isNext와 notNext에 대한 확률값 계산

## Fine-tuning

- 사전학습된 BERT를 각각의 NLP task에 맞춰 조정하는 과정
- BERT의 가중치를 기반으로, task-specific layer나 파라미터를 추가하여 labeled data에서 전이학습

![bert fine-tuning](/assets/images/blog/bertfinetune.png)

## 실험 결과

![bert experiment result 1](/assets/images/blog/bertscore01.png)

첫번째 실험은 GLUE benchmark dataset에서 평가한 결과입니다. GLUE는 general language understading task를 포함하고, BERT base, large 모두 기존 방법들보다 나은 성능을 보여주었고, large는 모든 task에서 나은 성능을 보였다.

![bert experiment result 2](/assets/images/blog/bertscore02.png)

두번째 실험은 **QA** dataset인 SQuAD에서 실험한 결과입니다. wikipedia의 context 지문을 이용하여 QA를 하는 데이터셋이다. Bert 모델들이 기존 방식보다 나은 성능을 보여주며, 가장 좋은 성능을 보인 모델을 앙상블한 모델과 QA dataset finetuning한 모델이 사람보다 뛰어난 성능을 보여주었다.

![bert experiment result 3](/assets/images/blog/bertscore03.png)

네번째 실험은 SWAG(Situations With Adversarial Generation) 데이터셋으로 이어지는 문장을 고르는 sentence pair inference task입니다. 이 실험 또한 Bert가 기존 모델보다 우수한 성능을 보였고, Bert largesms 전문가와 비슷한 성능을 보여주었다.

![bert experiment result 4](/assets/images/blog/bertscore.png)

모델 size에 따른 성능 비교로 모델의 size가 커질수록 성능이 향상된다는 것을 보인다.

## 한계점

bert는 양뱡향이지만 완전한 생성형 모델은 아니다. 그리고 pretrain-finetuning의 구조적 한계이다. GPT는 in-context learning으로 finetuning 없이 prompt만으로 downstream task 전환 가능이지만 Bert는 fine tuning이 필요하다. 

## 결론

Bert는 encoder 기반 양방향 모델으로 GPT와는 다르게 문장 해석, 형태소 인식 등등의 task에 특화되어, 양질의 pre-trained language representation를 얻을 수 있는 것과 동시에 down-stream task로의 손쉬운 fine-tuning이 가능한 bidirectional language model이다. 이 과정에서 pretraining의 MLM,NSP 방식이 효과적으로 적용되었고 다양한 task에서 Bert는 우수한 성능을 보여주었다. 

## 내생각

Bert의 양방향적 시도는 다른 모델들도 시도해볼만하지만 추론이 너무 느려진다는 문제를 해결하지 못하면 양방향성을 통한 모델은 상업적으로 사용하긴 힘들것이다. 그리고 Bert의 pretrain 기법을 통해 문맥, 문장 이해 성능을 끌어올리는 시도는 좋았다고 생각한다.
