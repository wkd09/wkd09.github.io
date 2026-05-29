---
layout: single
title: "공부"
permalink: /study/
author_profile: true
---

<section class="blog-intro">
  <p>AI, LLM, 딥러닝, 시스템 구현을 공부하며 정리하는 공간입니다.</p>
  <span>{{ site.categories.study.size }}개의 글</span>
</section>

{% assign study_posts = site.categories.study %}

{% if study_posts.size > 0 %}
<div class="post-list">
  {% for post in study_posts %}
    <article class="post-list__item">
      <div class="post-list__body">
        <a class="post-list__title-link" href="{{ post.url | relative_url }}">
          <h2>{{ post.title }}</h2>
        </a>
        <p class="post-list__meta">
          {{ post.date | date: "%Y.%m.%d" }}
          {% if post.tags.size > 0 %}
            · {{ post.tags | join: ", " }}
          {% endif %}
        </p>
        {% if post.excerpt %}
          <p>{{ post.excerpt | strip_html | truncate: 120 }}</p>
        {% endif %}
      </div>
    </article>
  {% endfor %}
</div>
{% else %}
<div class="empty-state">
  <p>아직 작성한 공부 글이 없습니다.</p>
  <span>공부 기록을 추가하면 이곳에 최신순으로 표시됩니다.</span>
</div>
{% endif %}
