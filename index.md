---
layout: single
author_profile: true
title: "전체글"
---

<section class="blog-intro">
  <p>AI, LLM, Agent, 시스템 구현을 공부하며 정리하는 공간입니다.</p>
  <span>{{ site.posts.size }}개의 글</span>
</section>

{% if site.posts.size > 0 %}
<div class="post-list">
  {% for post in site.posts %}
    <article class="post-list__item">
      <div class="post-list__body">
      <a class="post-list__title-link" href="{{ post.url | relative_url }}">
        <h2>{{ post.title }}</h2>
      </a>
      <p class="post-list__meta">
        {{ post.date | date: "%Y.%m.%d" }}
        {% if post.categories.size > 0 %}
          · {{ post.categories | join: ", " }}
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
  <p>아직 작성한 글이 없습니다.</p>
  <span>글을 추가하면 이곳에 최신순으로 표시됩니다.</span>
</div>
{% endif %}
