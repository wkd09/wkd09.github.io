---
title: "GraphRAG 논문 정리: 전체 문서 집합을 이해하는 RAG"
date: 2026-07-02 00:00:00 +0900
categories:
  - engineering
  - research
tags:
  - AI
  - LLM
  - RAG
  - GraphRAG
  - KnowledgeGraph
  - Paper
source: "arXiv:2404.16130"
---

# GraphRAG: 전체 문서 집합을 이해하는 RAG

이 글은 논문 [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)을 바탕으로 정리한 글이다.

> Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson  
> arXiv:2404.16130. [[Paper](https://arxiv.org/pdf/2404.16130)]

논문의 핵심 문제의식은 간단하다.

> 일반적인 RAG는 질문과 비슷한 chunk를 찾는 데는 강하지만, 문서 집합 전체를 이해해야 하는 질문에는 약하다.

예를 들어 다음 질문을 생각해보자.

```text
이 뉴스 데이터셋 전체에서 반복적으로 등장하는 공공보건 이슈는 무엇인가?
이 회사의 회의록 전체를 보면 조직의 주요 리스크는 어떻게 변화하고 있는가?
이 논문 묶음에서 AI 윤리와 관련된 주요 주제는 무엇이고, 서로 어떻게 연결되는가?
```

이런 질문은 답이 특정 문장 하나에 들어 있지 않다. 문서 전체에 흩어진 사건, 인물, 조직, 개념, 관계, 반복 패턴을 모아서 봐야 한다. 논문은 이런 질문을 `global sensemaking question`으로 본다. 즉, 단순 검색 문제가 아니라 **query-focused summarization**, 줄여서 `QFS` 문제에 가깝다.

GraphRAG는 이 문제를 해결하기 위해 문서 전체에서 지식 그래프를 만들고, 그래프의 커뮤니티를 요약한 뒤, 질문이 들어오면 커뮤니티 요약들을 다시 합쳐 답변한다.

한 줄로 요약하면 이렇다.

> GraphRAG는 `문서 -> 지식 그래프 -> 커뮤니티 요약 -> map-reduce 답변`으로 이어지는 global RAG 방식이다.

![GraphRAG architecture](/assets/images/blog/graphrag-architecture.png)

*GraphRAG 구현 아키텍처 예시. 논문의 본질은 특정 DB나 쿼리 언어가 아니라, 전체 corpus를 graph와 community summary로 구조화하는 데 있다.*

## 1. 기존 Vector RAG는 무엇을 잘하는가?

먼저 일반적인 RAG를 생각해보자. 가장 익숙한 구조는 다음과 같다.

```text
문서 수집
-> chunk 분할
-> embedding 생성
-> vector DB 저장
-> 질문 embedding
-> 유사한 chunk top-k 검색
-> LLM 답변
```

이 방식은 `local question`에 강하다. 답이 특정 문서 조각에 직접 들어 있는 질문이면 꽤 잘 작동한다.

예를 들면 이런 질문이다.

```text
Neo4j의 기본 Bolt 포트는 몇 번인가?
계약서에서 해지 통보 기간은 며칠인가?
논문에서 사용한 chunk size는 얼마인가?
어떤 회사가 2016년에 NeoChip을 인수했는가?
```

이런 질문은 질문과 의미적으로 가까운 chunk를 찾으면 된다. chunk 안에 답이 들어 있고, LLM은 그 chunk를 근거로 답을 만들면 된다.

하지만 질문이 조금만 넓어지면 문제가 생긴다.

```text
전체 문서에서 가장 중요한 주제는 무엇인가?
여러 문서에 걸쳐 반복되는 갈등 구조는 무엇인가?
이 corpus에서 기술 리더들은 규제를 어떻게 바라보는가?
데이터셋 전체를 보면 어떤 인물과 조직이 핵심 축인가?
```

이 질문들은 top-k chunk 몇 개만으로 답하기 어렵다. 왜냐하면 중요한 정보가 여러 문서에 분산되어 있기 때문이다. 검색 결과 상위 5개나 10개 chunk만 보면 자주 언급되는 일부 내용은 잡을 수 있지만, 전체 corpus의 구조는 놓칠 수 있다.

즉 일반 RAG의 한계는 단순히 검색 정확도가 낮다는 문제가 아니다. 더 근본적으로, **질문 자체가 검색 문제가 아니라 전체 요약 문제일 때** 기존 RAG 구조가 잘 맞지 않는다.

## 2. Local Question과 Global Question

이 논문을 이해하려면 `local question`과 `global question`을 구분하는 것이 중요하다.

| 구분 | 예시 | 필요한 능력 |
|---|---|---|
| Local question | "A 회사가 B 회사를 인수한 날짜는?" | 특정 근거 chunk 검색 |
| Global question | "이 데이터셋의 주요 테마는?" | 전체 corpus 구조 파악 |

Local question은 답이 좁다. 특정 문서, 특정 문단, 특정 문장에 답이 들어 있을 가능성이 높다.

반대로 global question은 답이 넓다. 여러 문서에 걸쳐 반복되는 패턴과 관계를 종합해야 한다. 이런 질문에 대해 vector search만 쓰면 다음 문제가 생긴다.

- 검색된 chunk가 전체 corpus를 대표하지 않을 수 있다.
- 많이 등장하지만 표현이 다양한 주제가 검색에서 빠질 수 있다.
- 서로 다른 문서 사이의 관계가 사라진다.
- 답변이 몇 개 사례에 과하게 의존할 수 있다.
- 긴 문서 집합을 매번 query time에 모두 요약하기에는 비용이 너무 크다.

GraphRAG는 이 문제를 query time 검색만으로 풀지 않는다. 대신 **index time에 corpus 전체를 미리 구조화**한다.

이 관점 전환이 논문의 핵심이다.

```text
기존 RAG:
질문이 들어온 뒤 관련 chunk를 찾는다.

GraphRAG:
질문이 들어오기 전에 corpus 전체를 graph와 summary로 구조화한다.
```

## 3. GraphRAG의 전체 파이프라인

논문에서 제안하는 GraphRAG의 큰 흐름은 다음과 같다.

```text
Source Documents
-> Text Chunks
-> Element Instances
-> Element Summaries
-> Graph Communities
-> Community Summaries
-> Community Answers
-> Global Answer
```

조금 더 실무적인 용어로 바꾸면 이렇게 볼 수 있다.

```text
문서
-> chunk
-> entity / relationship / claim 추출
-> knowledge graph 생성
-> community detection
-> community summary 생성
-> 질문별 partial answer 생성
-> final answer 생성
```

여기서 중요한 점은 GraphRAG가 단순히 "그래프 DB를 붙인 RAG"가 아니라는 것이다. 물론 Neo4j, FalkorDB, NetworkX, igraph 같은 그래프 저장소나 그래프 라이브러리를 쓸 수 있다. 하지만 논문이 말하는 핵심은 DB 제품이 아니라 **그래프 기반 요약 인덱스**다.

GraphRAG의 인덱스는 단순한 vector index가 아니다. 문서 안의 개체와 관계를 그래프로 만들고, 그 그래프를 community 단위로 나눈 뒤, 각 community가 무엇을 의미하는지 요약한 구조다.

## 4. Step 1: 문서를 Chunk로 나누기

첫 단계는 일반 RAG와 비슷하다. 문서를 chunk로 나눈다.

논문 실험에서는 약 600 token 단위 chunk와 100 token overlap을 사용했다. 이 값 자체가 절대 정답은 아니다. 중요한 것은 chunk 크기가 비용과 recall 사이의 trade-off라는 점이다.

chunk가 너무 작으면:

- LLM 호출 횟수가 늘어난다.
- 문맥이 부족해 entity와 relationship 추출 품질이 떨어질 수 있다.
- 같은 entity가 여러 chunk에 흩어져 중복 처리될 수 있다.

chunk가 너무 크면:

- LLM 호출 수는 줄어든다.
- 하지만 chunk 안의 세부 정보가 묻힐 수 있다.
- 긴 context에서 앞부분 정보가 덜 잘 활용될 수 있다.

일반 vector RAG에서도 chunk size는 중요하지만, GraphRAG에서는 더 중요하다. chunk는 단순 검색 단위가 아니라, 이후 그래프 추출의 입력이 되기 때문이다. chunk 설계가 나쁘면 그래프 자체가 왜곡된다.

## 5. Step 2: Entity, Relationship, Claim 추출

GraphRAG는 각 chunk에서 LLM을 사용해 구조화된 정보를 추출한다.

주로 추출하는 것은 세 가지다.

| 요소 | 의미 | 예시 |
|---|---|---|
| Entity | 사람, 조직, 장소, 개념, 사건 등 | `OpenAI`, `Sam Altman`, `AI regulation` |
| Relationship | entity 사이의 관계 | `Sam Altman - leads - OpenAI` |
| Claim | entity나 관계에 대한 검증 가능한 주장 | `OpenAI released GPT-4 in 2023` |

예를 들어 다음 문장이 있다고 하자.

```text
Quantum Systems acquired NeoChip in 2016 to expand its AI accelerator business.
```

LLM은 여기서 다음 구조를 추출할 수 있다.

```text
Entity:
- Quantum Systems
- NeoChip
- AI accelerator business

Relationship:
- Quantum Systems acquired NeoChip
- NeoChip is related to AI accelerator business

Claim:
- Quantum Systems acquired NeoChip in 2016.
```

이 단계는 GraphRAG의 품질을 크게 좌우한다. entity를 잘못 뽑거나 relationship을 이상하게 만들면, 이후 community detection과 summary도 함께 흔들린다.

그래서 실제 구현에서는 domain-specific prompt가 중요하다. 법률 문서라면 계약 당사자, 조항, 의무, 위반, 기간 같은 entity type이 중요할 수 있다. 의료 문서라면 질병, 증상, 약물, 검사, 부작용 같은 schema가 필요할 수 있다. 논문 요약 corpus라면 저자, 방법론, benchmark, metric, limitation 같은 type이 더 적절하다.

즉 GraphRAG는 "LLM에게 아무 entity나 뽑아줘"라고 맡기는 방식보다, **도메인에 맞는 entity schema와 few-shot 예시를 설계할수록** 더 실용적이다.

## 6. Step 3: Knowledge Graph 만들기

추출된 entity와 relationship은 knowledge graph로 저장된다.

기본 구조는 간단하다.

```text
Node = entity
Edge = relationship
Edge weight = 같은 관계가 발견된 횟수 또는 강도
Claim = node나 edge에 붙는 근거 정보
```

예를 들어 AI 연구 논문 corpus라면 다음 같은 그래프가 만들어질 수 있다.

![GraphRAG example graph](/assets/images/blog/graphrag-example-graph.png)

*지식 그래프 예시. 사람, 주제, 논문, 개념이 관계로 연결되고, 이 연결 구조를 통해 corpus 안의 주제 덩어리를 파악할 수 있다.*

그래프를 만들 때 실무적으로 가장 까다로운 문제 중 하나는 entity normalization이다.

예를 들어 다음 표현들은 같은 대상을 가리킬 수 있다.

```text
OpenAI
Open AI
OpenAI Inc.
the company behind ChatGPT
```

이 표현들을 모두 다른 node로 저장하면 그래프가 쪼개진다. 반대로 서로 다른 entity를 하나로 합치면 그래프가 오염된다.

논문 구현은 비교적 단순한 exact string matching을 사용했다고 설명한다. 하지만 실제 서비스에서는 더 정교한 entity resolution이 필요할 수 있다.

고려할 수 있는 방법은 다음과 같다.

- 표기 정규화: 대소문자, 공백, 약어 처리
- alias dictionary: 조직명, 인물명, 제품명 별칭 관리
- embedding 기반 유사도: 이름과 설명을 함께 비교
- LLM 기반 merge 판단: 두 entity가 같은 대상을 가리키는지 판정
- human-in-the-loop: 중요한 entity는 사람이 검수

GraphRAG에서 그래프는 단순 장식이 아니다. 이후 community detection과 summary의 기반이다. 그래서 entity resolution 품질은 전체 성능에 직접적인 영향을 준다.

## 7. Step 4: Graph Community Detection

Knowledge graph가 만들어지면, 다음 단계는 graph community detection이다.

Community detection은 그래프에서 서로 강하게 연결된 node 그룹을 찾는 과정이다. 논문은 Leiden 알고리즘을 사용한다.

쉽게 말하면 그래프 안에서 자연스럽게 뭉쳐 있는 주제 덩어리를 찾는 것이다.

예를 들어 뉴스 corpus라면 다음 같은 community가 생길 수 있다.

```text
Community 1:
공공보건, 백신, 병원, 질병관리청, 감염병

Community 2:
반도체, 수출 규제, 공급망, 삼성전자, 대만

Community 3:
교육 정책, 입시, 학교, 교사, 학부모
```

논문 corpus처럼 팟캐스트나 뉴스 데이터라면 community는 특정 인물, 조직, 주제, 사건을 중심으로 형성될 수 있다.

이 단계가 중요한 이유는 전체 corpus를 한 번에 요약하지 않기 위해서다. corpus가 크면 원문 전체를 LLM context에 넣을 수 없다. 그렇다고 무작위 chunk를 요약하면 전체 구조가 깨진다. GraphRAG는 그래프 구조를 이용해 의미 있게 연결된 덩어리부터 요약한다.

즉 community는 단순 클러스터가 아니라, 이후 global summary를 만들기 위한 중간 단위다.

## 8. Step 5: Community Summary 만들기

각 community가 만들어지면, GraphRAG는 해당 community의 entity, relationship, claim을 바탕으로 LLM에게 summary를 생성하게 한다.

이 summary는 일반적인 chunk summary와 다르다.

chunk summary는 특정 텍스트 조각을 요약한다.

```text
이 문단은 A 회사가 B 회사를 인수한 사건을 설명한다.
```

community summary는 그래프 안의 연결 구조를 요약한다.

```text
이 community는 AI 규제와 기업 거버넌스를 중심으로 형성되어 있다.
주요 entity는 OpenAI, Microsoft, EU AI Act, model safety이며,
핵심 관계는 기업의 모델 출시와 규제 기관의 감독 강화 사이의 긴장이다.
```

이런 community summary는 GraphRAG에서 일종의 `global memory index` 역할을 한다. query time에 원문 전체를 다시 읽지 않고, 미리 만들어둔 community summary를 활용할 수 있기 때문이다.

여기서 GraphRAG의 비용 구조가 드러난다.

- index time 비용은 커진다.
- query time에서는 압축된 summary를 재사용할 수 있다.

한 번만 질문할 corpus라면 이 비용이 부담스럽다. 하지만 같은 corpus에 대해 여러 global question을 반복해서 물어본다면 community summary는 꽤 유용한 인덱스가 된다.

## 9. Query Time: Map-Reduce로 답변 만들기

GraphRAG의 query time 과정은 map-reduce에 가깝다.

질문이 들어오면 GraphRAG는 여러 community summary를 사용해 partial answer를 만든다. 그 뒤 partial answer들을 다시 합쳐 final answer를 만든다.

흐름은 다음과 같다.

```text
User question
-> community summary 여러 개 선택 또는 사용
-> 각 summary에 대해 partial answer 생성
-> partial answer에 helpfulness score 부여
-> score가 낮은 답변 제거
-> score가 높은 답변부터 context에 넣기
-> final global answer 생성
```

논문에서는 map 단계에서 LLM이 각 community summary를 보고 질문에 대한 intermediate answer를 생성한다. 동시에 이 answer가 질문에 얼마나 도움이 되는지 0부터 100 사이의 score를 매긴다. score가 0인 답변은 제거하고, reduce 단계에서는 score가 높은 답변을 우선적으로 사용한다.

이 방식의 장점은 global question을 다룰 때 답변 후보를 넓게 만들 수 있다는 것이다. 하나의 검색 결과에 모든 것을 걸지 않고, 여러 community에서 부분 답변을 만든 뒤 합친다.

예를 들어 질문이 다음과 같다고 하자.

```text
이 AI 윤리 corpus에서 반복적으로 등장하는 핵심 쟁점은 무엇인가?
```

그러면 각 community는 다른 관점을 줄 수 있다.

```text
Community A:
모델 편향과 공정성 문제

Community B:
투명성과 설명 가능성 문제

Community C:
데이터 프라이버시와 동의 문제

Community D:
규제, 거버넌스, 책임 소재 문제
```

최종 답변은 이 partial answer들을 합쳐 corpus 전체의 주요 쟁점을 정리한다.

이것이 GraphRAG가 vector RAG보다 global sensemaking question에서 강한 이유다. top-k chunk 몇 개를 보는 대신, corpus 전체에서 만들어진 여러 community summary를 활용하기 때문이다.

## 10. 실험 설정

논문은 두 개의 약 100만 token 규모 corpus에서 GraphRAG를 평가했다.

첫 번째는 팟캐스트 transcript 데이터셋이다.

| 항목 | 내용 |
|---|---|
| 데이터 | Behind the Tech with Kevin Scott transcript |
| 규모 | 약 100만 token |
| chunk 수 | 1,669개 |
| chunk size | 600 token |
| overlap | 100 token |

두 번째는 뉴스 기사 데이터셋이다.

| 항목 | 내용 |
|---|---|
| 데이터 | 2013년 9월부터 2023년 12월까지의 뉴스 기사 |
| 규모 | 약 170만 token |
| chunk 수 | 3,197개 |
| chunk size | 600 token |
| overlap | 100 token |

비교 조건은 크게 세 종류로 볼 수 있다.

| 방식 | 의미 |
|---|---|
| GraphRAG community summary | 그래프 community level별 요약 사용 |
| TS | source text를 map-reduce로 직접 요약 |
| SS | semantic search 기반 vector RAG baseline |

GraphRAG community summary는 다시 여러 level로 나뉜다.

| 조건 | 의미 |
|---|---|
| C0 | root-level community summaries |
| C1 | high-level community summaries |
| C2 | intermediate-level community summaries |
| C3 | low-level community summaries |

여기서 level이 낮아질수록 더 세부적인 community를 많이 사용한다고 볼 수 있다. 더 자세한 정보를 담을 수 있지만 query token 비용도 커진다.

## 11. 평가 기준

global sensemaking question은 정답이 하나로 딱 떨어지지 않는다.

예를 들어 "이 데이터셋의 주요 테마는 무엇인가?"라는 질문에는 여러 좋은 답이 가능하다. 그래서 exact match accuracy 같은 metric을 쓰기 어렵다.

논문은 LLM-as-a-judge 방식으로 답변을 비교한다. 평가 기준은 네 가지다.

| 기준 | 의미 |
|---|---|
| Comprehensiveness | 질문의 여러 측면을 충분히 다루는가 |
| Diversity | 다양한 관점과 insight를 제공하는가 |
| Empowerment | 사용자가 주제를 이해하고 판단하는 데 도움이 되는가 |
| Directness | 질문에 직접적이고 명확하게 답하는가 |

여기서 `Directness`는 약간 특이하다. GraphRAG는 더 포괄적이고 다양한 답변을 만들 수 있지만, 그러다 보면 답이 길어지고 덜 직접적으로 느껴질 수 있다. 그래서 directness는 일종의 control criterion으로 볼 수 있다.

즉 논문이 보고 싶은 것은 단순히 "어느 답변이 더 길고 풍부한가"가 아니다. global question에서 더 넓고 다양한 정보를 담으면서도 질문에 답하고 있는지를 비교한다.

## 12. 주요 결과

핵심 결과는 명확하다.

> GraphRAG 계열 global approach는 semantic search 기반 vector RAG보다 comprehensiveness와 diversity에서 크게 우수했다.

논문에 따르면 Podcast 데이터셋에서는 GraphRAG 방식이 semantic search baseline보다 comprehensiveness와 diversity에서 높은 win rate를 보였다. News 데이터셋에서도 비슷하게 global question에 대해 더 포괄적이고 다양한 답변을 생성했다.

특히 중요한 해석은 비용과 품질의 trade-off다.

| 방식 | 특징 |
|---|---|
| C3 | 세부 community를 많이 사용해 정보량이 많지만 token 비용이 큼 |
| TS | 원문을 map-reduce로 직접 요약해 강하지만 비용이 큼 |
| C0 | root-level summary만 사용해 비용이 낮고 재사용성이 좋음 |
| SS | 비용은 낮지만 global question에서 정보가 부족할 수 있음 |

논문에서 흥미로운 지점은 C0다. root-level community summary는 매우 압축된 인덱스인데도, global question에서 semantic search baseline보다 좋은 결과를 보인다. 게다가 query token 비용은 훨씬 낮다.

이 말은 GraphRAG가 항상 더 많은 token을 써서 이기는 방식이 아니라는 뜻이다. index time에 community summary를 잘 만들어두면, query time에는 꽤 압축된 context만으로도 global answer를 만들 수 있다.

## 13. GraphRAG의 핵심은 Graph DB가 아니다

GraphRAG라는 이름 때문에 자주 생기는 오해가 있다.

```text
GraphRAG = Neo4j 같은 graph DB에 문서를 넣고 Cypher로 검색하는 RAG
```

이 설명은 일부 구현에는 맞을 수 있지만, 논문 전체를 설명하기에는 부족하다.

논문의 핵심은 다음에 가깝다.

```text
GraphRAG = 전체 corpus를 entity graph로 구조화하고,
graph community를 요약해,
global question에 답하는 hierarchical summarization index
```

물론 그래프 DB는 유용하다. entity와 relationship을 저장하고, 특정 subgraph를 탐색하고, Cypher나 Gremlin 같은 query language로 검색할 수 있다. 하지만 GraphRAG 논문에서 더 중요한 것은 "그래프를 저장했다"가 아니라 "그래프 community를 미리 요약했다"이다.

그래서 GraphRAG를 구현할 때는 다음 질문을 먼저 해야 한다.

```text
어떤 graph DB를 쓸까?
```

보다 더 중요한 질문은 이것이다.

```text
이 corpus에서 어떤 entity와 relationship이 의미 있는가?
어떤 community 구조가 global question에 도움이 되는가?
community summary를 어떤 형식으로 만들 것인가?
query time에 어떤 summary를 어떻게 합칠 것인가?
```

DB 선택은 그 다음 문제다.

## 14. Vector RAG와 GraphRAG 비교

두 방식을 단순 비교하면 다음과 같다.

| 항목 | Vector RAG | GraphRAG |
|---|---|---|
| 기본 단위 | chunk embedding | entity, relationship, community summary |
| 강한 질문 | local fact question | global sensemaking question |
| 인덱싱 비용 | 상대적으로 낮음 | 높음 |
| query 비용 | top-k 기준 낮음 | 설정에 따라 다름 |
| 구조 정보 | 약함 | 강함 |
| 반복 질의 | corpus 구조 재사용 제한적 | community summary 재사용 가능 |
| 주요 리스크 | 관련 chunk 누락 | 잘못된 entity/relationship 추출 |

중요한 것은 둘 중 하나만 고르는 문제가 아니라는 점이다. 실무에서는 두 방식을 함께 쓰는 것이 자연스럽다.

예를 들어 다음처럼 나눌 수 있다.

```text
질문: "계약서에서 해지 통보 기간은?"
-> Vector RAG로 관련 조항 검색

질문: "전체 계약서 묶음에서 반복적으로 나타나는 리스크는?"
-> GraphRAG community summary 활용
```

또는 GraphRAG 답변 중 특정 claim에 대해 원문 근거를 확인할 때 vector search를 다시 사용할 수도 있다.

GraphRAG는 vector RAG를 대체한다기보다, 기존 RAG가 약한 global question 영역을 보완하는 방식으로 보는 것이 좋다.

## 15. 구현한다면 어떻게 시작할까?

논문 전체를 그대로 구현하려고 하면 꽤 무겁다. 처음에는 작게 줄여 시작하는 것이 좋다.

가장 작은 구현 흐름은 다음과 같다.

```text
1. 문서 20~100개를 준비한다.
2. 문서를 500~800 token 단위로 chunking한다.
3. 각 chunk에서 entity와 relationship을 JSON으로 추출한다.
4. entity normalization을 간단히 수행한다.
5. NetworkX나 Neo4j에 graph를 만든다.
6. Louvain 또는 Leiden으로 community를 찾는다.
7. community별 summary를 생성한다.
8. 질문이 들어오면 summary별 partial answer를 만든다.
9. partial answer를 합쳐 final answer를 생성한다.
```

초기 구현에서는 graph DB보다 파일 기반으로 시작해도 된다.

```text
chunks.jsonl
entities.jsonl
relationships.jsonl
communities.json
community_summaries.json
```

이 구조만으로도 GraphRAG의 핵심 아이디어를 실험할 수 있다. 이후 데이터가 커지고 query가 복잡해지면 Neo4j, FalkorDB, PostgreSQL + Apache AGE, NetworkX, igraph 같은 선택지를 검토하면 된다.

## 16. Entity Extraction Prompt 예시

실제 구현에서 중요한 부분은 추출 prompt다. 예를 들어 논문 corpus를 대상으로 한다면 다음과 같은 JSON schema를 요구할 수 있다.

```text
You are extracting a knowledge graph from an AI research paper chunk.

Extract:
1. entities: method, model, dataset, benchmark, metric, organization, author, task
2. relationships: proposes, improves, evaluates_on, uses, compares_with, reports_metric
3. claims: factual statements supported by the chunk

Return JSON only.
```

출력은 이런 형식으로 제한할 수 있다.

```json
{
  "entities": [
    {
      "name": "GraphRAG",
      "type": "method",
      "description": "A graph-based approach to question answering over private text corpora."
    }
  ],
  "relationships": [
    {
      "source": "GraphRAG",
      "target": "Query-Focused Summarization",
      "type": "addresses",
      "description": "GraphRAG addresses global sensemaking questions as a QFS task."
    }
  ],
  "claims": [
    {
      "subject": "GraphRAG",
      "claim": "GraphRAG improves comprehensiveness and diversity over a semantic search baseline for global questions."
    }
  ]
}
```

이때 중요한 것은 자유로운 자연어 출력보다 구조화된 JSON을 받는 것이다. 그래야 후처리, validation, graph loading이 쉬워진다.

## 17. Community Summary Prompt 예시

community summary를 만들 때는 단순히 entity 목록만 던지는 것보다, node, edge, claim을 함께 제공하는 것이 좋다.

예를 들어 prompt는 다음처럼 구성할 수 있다.

```text
You are summarizing a graph community extracted from a document corpus.

Given:
- entities
- relationships
- claims
- source references

Write a concise report that includes:
1. the main topic of this community
2. the most important entities
3. the most important relationships
4. recurring patterns or tensions
5. why this community may be useful for answering global questions
```

출력도 일정한 형식으로 유지하는 것이 좋다.

```text
Title:
Main topic:
Key entities:
Key relationships:
Important claims:
Potential relevance:
```

이렇게 만들어두면 query time에 partial answer를 만들 때 훨씬 안정적이다.

## 18. 정형 데이터에는 그대로 쓰면 안 된다

GraphRAG 논문은 기본적으로 비정형 텍스트 corpus를 대상으로 한다. 그래서 LLM으로 entity와 relationship을 추출하는 과정이 중요하다.

하지만 CSV, DB table, 공공데이터 같은 정형 데이터라면 이야기가 달라진다.

예를 들어 지하철 승하차 데이터가 있다고 하자.

```text
날짜, 역명, 노선, 승차 인원, 하차 인원
```

이 경우에는 LLM에게 entity를 뽑으라고 시키는 것보다, 규칙 기반으로 graph를 만드는 편이 더 정확하다.

```text
Node:
- Station
- Line
- Date
- Region

Edge:
- Station BELONGS_TO Line
- Station HAS_RIDERSHIP Record
- Station LOCATED_IN Region
- Station SIMILAR_PATTERN_TO Station
```

정형 데이터에서는 LLM extraction이 오히려 오류를 만들 수 있다. 이미 schema가 있기 때문이다. 이런 경우 GraphRAG의 아이디어 중 가져올 것은 "LLM extraction"이 아니라 "community와 summary를 이용해 global question에 답한다"는 부분이다.

즉 데이터 유형에 따라 구현 방식은 달라져야 한다.

| 데이터 유형 | 추천 방식 |
|---|---|
| 뉴스, 회의록, 논문, 보고서 | LLM 기반 entity/relationship extraction |
| CSV, DB, 로그, 공공데이터 | schema 기반 graph construction |
| 혼합 데이터 | 구조화 가능한 부분은 규칙 기반, 비정형 부분은 LLM 기반 |

## 19. 한계와 주의점

GraphRAG는 강력하지만 만능은 아니다. 논문을 읽을 때 특히 조심해야 할 부분이 있다.

### 19.1 인덱싱 비용이 크다

일반 vector RAG는 chunk embedding을 만들면 된다. GraphRAG는 여기에 entity extraction, relationship extraction, claim extraction, graph construction, community detection, community summary 생성이 추가된다.

즉 초기 비용이 크다.

문서가 계속 바뀌는 corpus라면 incremental update도 고민해야 한다. 새 문서가 들어올 때마다 전체 graph를 다시 만들 것인지, 일부만 갱신할 것인지가 문제가 된다.

### 19.2 추출 품질에 크게 의존한다

LLM이 entity나 relationship을 잘못 추출하면 그래프가 오염된다. 특히 전문 도메인에서는 일반 LLM이 중요한 entity를 놓치거나, 관계를 너무 느슨하게 만들 수 있다.

예를 들어 법률 문서에서 `obligation`, `liability`, `termination`, `jurisdiction` 같은 개념을 정확히 구분하지 못하면 summary가 쓸모없어진다.

### 19.3 Entity resolution이 어렵다

같은 entity를 하나로 합치고, 다른 entity를 분리하는 것은 생각보다 어렵다. 특히 사람 이름, 조직명, 약어, 제품명, 번역명이 섞이면 그래프 품질이 빠르게 흔들린다.

### 19.4 환각이 자동으로 해결되는 것은 아니다

GraphRAG가 더 포괄적인 답변을 만든다고 해서 hallucination이 자동으로 줄어든다고 말할 수는 없다.

논문도 fabrication rate 비교는 향후 연구 과제로 남긴다. 따라서 정확한 표현은 다음에 가깝다.

```text
GraphRAG는 global question에서 더 포괄적이고 다양한 답변을 생성했다.
```

이지,

```text
GraphRAG는 환각을 확실히 줄인다.
```

가 아니다.

실무에서는 반드시 source attribution, claim verification, 원문 근거 연결을 함께 설계해야 한다.

### 19.5 모든 질문에 GraphRAG가 필요한 것은 아니다

단순 사실 질문에는 vector RAG가 더 싸고 빠르다. GraphRAG를 모든 질문에 쓰면 비용과 지연 시간이 불필요하게 커질 수 있다.

그래서 실제 시스템에서는 query router가 필요할 수 있다.

```text
local fact question -> vector search
global sensemaking question -> GraphRAG
entity relationship question -> graph query
```

## 20. 실무 시스템으로 설계하기

GraphRAG를 서비스에 넣는다면 다음처럼 구성할 수 있다.

```text
Ingestion Pipeline
1. 문서 수집
2. chunking
3. entity / relationship / claim extraction
4. entity normalization
5. graph 저장
6. community detection
7. community summary 생성
8. vector index와 graph index 동시 저장

Query Pipeline
1. 질문 분류
2. local question이면 vector RAG
3. global question이면 community summary 기반 map-reduce
4. entity path question이면 graph traversal
5. 최종 답변에 source와 confidence 제공
```

여기서 운영상 중요한 것은 observability다.

다음 항목을 로그로 남기는 것이 좋다.

- 어떤 community summary가 사용되었는가
- partial answer score는 얼마였는가
- final answer가 어떤 partial answer를 근거로 삼았는가
- 답변에 포함된 claim이 어떤 원문 chunk에서 왔는가
- entity merge가 어떻게 일어났는가

GraphRAG는 중간 단계가 많다. 그래서 디버깅이 어렵다. 처음부터 중간 산출물을 저장하고 확인할 수 있게 만드는 것이 좋다.

## 21. 이 논문의 의의

이 논문의 의의는 "RAG에 그래프를 붙였다"가 아니다.

더 정확히는 RAG 문제를 다음처럼 확장했다는 데 있다.

```text
기존 RAG:
질문과 관련된 문서를 찾아 답한다.

GraphRAG:
문서 집합 전체의 구조를 미리 만들고,
그 구조를 이용해 전체 corpus에 대한 질문에 답한다.
```

즉 GraphRAG는 retrieval을 넘어 corpus understanding에 가까운 방향으로 간다.

이 관점은 중요하다. 앞으로 기업 내부 문서, 연구 논문 묶음, 고객 상담 로그, 회의록, 뉴스 아카이브 같은 데이터를 다룰 때 사용자가 원하는 질문은 단순히 "어느 문서에 뭐라고 쓰여 있나?"가 아닐 가능성이 크다.

사용자는 더 자주 이렇게 묻는다.

```text
전체적으로 무슨 일이 일어나고 있나?
핵심 리스크는 무엇인가?
반복되는 패턴은 무엇인가?
서로 연결된 이슈는 무엇인가?
어떤 주제가 시간이 지나며 중요해지고 있는가?
```

이런 질문에 답하려면 top-k retrieval만으로는 부족하다. corpus 전체를 구조화하고, 요약하고, 다시 질문에 맞게 재조합해야 한다.

GraphRAG는 그 방향을 보여준 대표적인 논문이다.

## 정리

GraphRAG의 핵심은 다음과 같다.

```text
문서 전체에서 entity와 relationship을 추출한다.
그 결과로 knowledge graph를 만든다.
그래프에서 community를 찾는다.
각 community를 미리 요약한다.
질문이 들어오면 community summary들로 partial answer를 만들고,
그 답들을 다시 합쳐 global answer를 만든다.
```

Vector RAG가 local fact retrieval에 강하다면, GraphRAG는 global sensemaking question에 강하다.

다만 비용, entity extraction 품질, entity resolution, hallucination 검증 문제는 여전히 중요하다. 그래서 실무에서는 GraphRAG를 vector RAG의 대체재로 보기보다, global question을 위한 보완 레이어로 설계하는 편이 좋다.

## 한 줄 요약

GraphRAG는 그래프 DB를 붙인 RAG가 아니라, 전체 문서 집합을 이해하기 위해 knowledge graph와 community summary를 만드는 global summarization index다.
