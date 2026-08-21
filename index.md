---
layout: single
author_profile: true
title: "AI, LLM, Agent를 기록합니다."
---

<section class="blog-intro">
  <span>{{ site.posts.size }}개의 글</span>
</section>

{% assign study_entry = site.categories.study | first %}
{% assign research_entry = site.categories.research | first %}
{% assign engineering_entry = site.categories.engineering | first %}

<section class="reading-paths" aria-labelledby="reading-paths-title">
  <div class="reading-paths__heading">
    <p>탐색하기</p>
    <h2 id="reading-paths-title">관심 있는 흐름부터 시작하세요.</h2>
  </div>

  <div class="reading-paths__grid">
    <a class="reading-path reading-path--study" href="{{ '/study/' | relative_url }}">
      <span class="reading-path__eyebrow">공부</span>
      <strong>읽기 시작</strong>
      <span>AI와 딥러닝의 기본 개념부터 차근차근 살펴봅니다.</span>
      {% if study_entry %}<small>최근: {{ study_entry.title }}</small>{% endif %}
    </a>

    <a class="reading-path reading-path--research" href="{{ '/research/' | relative_url }}">
      <span class="reading-path__eyebrow">논문</span>
      <strong>최근 논문</strong>
      <span>모델 구조와 최신 연구를 맥락까지 함께 정리합니다.</span>
      {% if research_entry %}<small>최근: {{ research_entry.title }}</small>{% endif %}
    </a>

    <a class="reading-path reading-path--engineering" href="{{ '/engineering/' | relative_url }}">
      <span class="reading-path__eyebrow">구현</span>
      <strong>구현 노트</strong>
      <span>학습, 서빙, 최적화에서 마주친 실제 병목을 기록합니다.</span>
      {% if engineering_entry %}<small>최근: {{ engineering_entry.title }}</small>{% endif %}
    </a>
  </div>
</section>

<section class="latest-notes" aria-labelledby="latest-notes-title">
  <div class="latest-notes__heading">
    <h2 id="latest-notes-title">최근 글</h2>
  </div>

{% if site.posts.size > 0 %}
<div class="post-list">
  {% for post in site.posts %}
    {% include post-list-item.html post=post %}
  {% endfor %}
</div>
{% else %}
<div class="empty-state">
  <p>아직 작성한 글이 없습니다.</p>
  <span>글을 추가하면 이곳에 최신순으로 표시됩니다.</span>
</div>
{% endif %}
</section>
