---
title: "구현 / 프로그래밍"
permalink: /engineering/
layout: single
author_profile: true
---

<section class="blog-intro">
  <p>구현, 시스템 설계, 성능 최적화, LLM 서빙을 정리하는 공간입니다.</p>
  <span>{{ site.categories.engineering.size }}개의 글</span>
</section>

{% assign engineering_posts = site.categories.engineering %}

{% if engineering_posts.size > 0 %}
<div class="post-list">
  {% for post in engineering_posts %}
    {% include post-list-item.html post=post %}
  {% endfor %}
</div>
{% else %}
<div class="empty-state">
  <p>아직 작성한 구현 글이 없습니다.</p>
  <span>구현 기록을 추가하면 이곳에 최신순으로 표시됩니다.</span>
</div>
{% endif %}
