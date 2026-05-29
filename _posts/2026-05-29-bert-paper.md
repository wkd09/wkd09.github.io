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

BERT는 **Bidirectional Encoder Representations from Transformers**의 약자다.

기존 언어 모델은 주로 왼쪽에서 오른쪽으로 읽는 단방향 언어 모델이었다. GPT 계열처럼 다음 token을 예측하는 모델은 self-attention에서 이전 token만 볼 수 있다.

BERT는 Transformer encoder를 사용해 왼쪽과 오른쪽 context를 모두 보는 양방향 표현을 학습한다.

## 연구 배경

질문 응답, 문장 관계 판단, 개체명 인식 같은 자연어 이해 task에서는 단방향 context만으로 부족할 수 있다.

예를 들어 문장 중간의 단어 의미를 이해하려면 앞 단어뿐 아니라 뒤 단어도 중요하다. BERT는 이런 문제를 해결하기 위해 bidirectional pre-training을 사용한다.

## BERT의 핵심 아이디어

BERT의 핵심은 두 가지 pre-training task다.

- Masked Language Model
- Next Sentence Prediction

BERT는 생성형 모델이라기보다 언어 이해를 위한 encoder 기반 모델이다.

## Embedding

BERT 입력 embedding은 세 가지를 더해 만든다.

## Token Embedding

BERT는 WordPiece tokenization을 사용한다.

단어를 그대로 쓰는 것이 아니라 subword 단위로 나눈다. 자주 등장하는 단어는 그대로 vocabulary에 들어가고, 드문 단어는 subword로 분할된다.

이 방식은 OOV 문제를 줄이고, 다양한 단어 형태를 처리하는 데 유리하다.

## Segment Embedding

BERT는 두 문장을 함께 입력받을 수 있다.

문장 A와 문장 B를 구분하기 위해 `[SEP]` token을 넣고, 각 token이 어느 문장에 속하는지 segment embedding으로 표시한다.

## Position Embedding

Transformer encoder는 recurrence가 없기 때문에 token 순서 정보를 따로 넣어야 한다.

BERT는 sin/cos positional encoding이 아니라 학습 가능한 position embedding을 사용한다. 일반적으로 최대 sequence length 512에 맞춰 위치 embedding을 학습한다.

## Masked Language Model

MLM은 입력 token 중 일부를 가리고, 주변 context를 이용해 가려진 token을 맞히는 task다.

BERT는 입력 token의 15%를 선택한다. 이 중:

- 80%는 `[MASK]`로 바꾼다.
- 10%는 random token으로 바꾼다.
- 10%는 그대로 둔다.

이렇게 하는 이유는 pre-training과 fine-tuning 사이의 불일치를 줄이기 위해서다. 실제 fine-tuning 입력에는 `[MASK]` token이 거의 등장하지 않기 때문이다.

## Next Sentence Prediction

NSP는 두 문장이 실제로 이어지는 문장인지 예측하는 task다.

데이터는 다음처럼 구성한다.

- 50%: 실제로 이어지는 문장, `IsNext`
- 50%: 무작위로 뽑은 관련 없는 문장, `NotNext`

`[CLS]` token의 final hidden vector를 사용해 두 문장의 관계를 분류한다.

## Fine-tuning

BERT는 pre-training 후 downstream task에 맞춰 fine-tuning한다.

분류 task에서는 `[CLS]` token representation 위에 classifier를 붙일 수 있고, token-level task에서는 각 token representation을 사용한다.

중요한 점은 task-specific layer를 조금 추가하더라도, BERT 전체 parameter를 downstream data로 함께 조정한다는 것이다.

## 실험 결과와 의미

BERT는 GLUE, SQuAD, SWAG 같은 다양한 자연어 이해 benchmark에서 강한 성능을 보였다.

특히 BERT-large는 여러 task에서 기존 방법보다 좋은 결과를 냈다. 이는 양방향 pre-training과 Transformer encoder representation이 자연어 이해 task에 효과적이라는 것을 보여준다.

## 한계

BERT는 양방향 encoder 기반 모델이기 때문에 GPT처럼 자연스럽게 긴 텍스트를 autoregressive하게 생성하는 모델은 아니다.

또한 downstream task마다 fine-tuning이 필요하다. GPT 계열의 in-context learning처럼 prompt만으로 task를 전환하는 방식과는 다르다.

## 정리

BERT의 핵심은 다음과 같다.

- Transformer encoder 기반 양방향 언어 표현 학습
- MLM으로 양방향 context를 활용한 token 예측
- NSP로 문장 관계 학습
- pre-training 후 downstream task fine-tuning

BERT는 생성보다 언어 이해에 초점이 맞춰진 모델이다. 문장 분류, QA, token classification 같은 task에서 encoder representation의 강점을 잘 보여준다.

## 참고

- Paper: <https://arxiv.org/pdf/1810.04805>
